import lightning as L
import torch
import os
from os import path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from xreflection.utils.registry import MODEL_REGISTRY
from xreflection import build_network, build_loss
from xreflection.metrics import calculate_metric
from xreflection.utils import imwrite, tensor2img
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info, rank_zero_warn
from torchmetrics import MetricCollection


@MODEL_REGISTRY.register()
class ClsModel(L.LightningModule):
    """Classification Module for reflection removal using PyTorch Lightning.
    
    This module implements a classification-based approach for single image reflection removal.
    It supports progressive multi-scale image processing, EMA model updates,
    and extensive validation metrics.
    """

    def __init__(self, opt):
        """Initialize the ClsModel.
        
        Args:
            opt (dict): Configuration options.
        """
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.is_train = opt['is_train']

        # Define network
        self.net_g = build_network(opt['network_g'])

        # Losses (initialized in setup)
        self.cri_pix = None
        self.cri_perceptual = None
        self.cri_grad = None

        # Initialize metric tracking
        self.best_metric_results = {}
        self.current_val_metrics = {}

        # Flag to indicate if using EMA - will be set by EMACallback
        self.use_ema = False

    def setup(self, stage: Optional[str] = None):
        """Setup module based on stage.
        
        Args:
            stage (str, optional): 'fit', 'validate', 'test', or 'predict'
        """
        self.print_network()

        # Load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_weights(load_path)

        if stage == 'fit' or stage is None:
            self.setup_losses()

    @rank_zero_only
    def print_network(self):
        """Print network information"""
        net_params = sum(map(lambda x: x.numel(), self.net_g.parameters()))
        rank_zero_info(f'Network: {self.net_g.__class__.__name__}, with parameters: {net_params:,d}')

    def load_weights(self, load_path):
        """Load pretrained weights.
        
        Args:
            load_path (str): Path to the checkpoint file.
        """
        param_key = self.opt['path'].get('param_key_g', 'params')
        strict_load = self.opt['path'].get('strict_load_g', True)

        if self.trainer is None or self.trainer.global_rank == 0:
            rank_zero_info(f'Loading weights from {load_path} with param key: [{param_key}]')

        # Load weights
        checkpoint = torch.load(load_path, map_location='cpu')

        # Check available keys in checkpoint for better debugging
        if self.trainer is None or self.trainer.global_rank == 0:
            if isinstance(checkpoint, dict):
                rank_zero_info(f"Available keys in checkpoint: {list(checkpoint.keys())}")

        # Try to load with specified param_key, then fallback to alternatives
        if param_key in checkpoint:
            weights = checkpoint[param_key]
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"Successfully loaded weights using key '{param_key}'")
        elif 'params_ema' in checkpoint and param_key != 'params_ema':
            weights = checkpoint['params_ema']
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"Key '{param_key}' not found, using 'params_ema' instead")
        elif 'params' in checkpoint and param_key != 'params':
            weights = checkpoint['params']
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"Key '{param_key}' not found, using 'params' instead")
        else:
            # If no recognized keys, use the entire checkpoint
            weights = checkpoint
            if self.trainer is None or self.trainer.global_rank == 0:
                rank_zero_info(f"No recognized parameter keys found, using entire checkpoint")

        # Remove unnecessary 'module.' prefix
        for k, v in list(weights.items()):
            if k.startswith('module.'):
                weights[k[7:]] = weights.pop(k)

        # Load to model
        self._print_different_keys_loading(self.net_g, weights, strict_load)
        self.net_g.load_state_dict(weights, strict=strict_load)

    @rank_zero_only
    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print key differences when loading models.
        
        Args:
            crt_net (nn.Module): Current network.
            load_net (dict): Loaded network state dict.
            strict (bool): Whether to strictly enforce parameter shapes.
        """
        # Get network state dict
        crt_net_keys = set(crt_net.state_dict().keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            rank_zero_warn('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                rank_zero_warn(f'  {v}')
            rank_zero_warn('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                rank_zero_warn(f'  {v}')

        # Check sizes of the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net.state_dict()[k].size() != load_net[k].size():
                    rank_zero_warn(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net.state_dict()[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def setup_losses(self):
        """Setup loss functions"""
        if not hasattr(self, 'cri_pix') or self.cri_pix is None:
            if self.opt['train'].get('pixel_opt'):
                self.cri_pix = build_loss(self.opt['train']['pixel_opt'])

        if not hasattr(self, 'cri_perceptual') or self.cri_perceptual is None:
            if self.opt['train'].get('perceptual_opt'):
                self.cri_perceptual = build_loss(self.opt['train']['perceptual_opt'])

        if not hasattr(self, 'cri_grad') or self.cri_grad is None:
            if self.opt['train'].get('grad_opt'):
                self.cri_grad = build_loss(self.opt['train']['grad_opt'])

    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            tuple: Classification outputs and image outputs.
        """
        return self.net_g(x)

    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch (dict): Input batch containing 'input', 'target_t', 'target_r'.
            batch_idx (int): Batch index.
            
        Returns:
            torch.Tensor: Total loss.
        """
        # Get inputs
        inp = batch['input']
        target_t = batch['target_t']
        target_r = batch['target_r']

        # Forward pass
        x_cls_out, x_img_out = self.net_g(inp)
        output_clean, output_reflection = x_img_out[-1][:, :3, ...], x_img_out[-1][:, 3:, ...]

        # Calculate losses
        loss_dict = OrderedDict()
        pix_t_loss_list = []
        pix_r_loss_list = []
        per_loss_list = []
        grad_loss_list = []

        for out_imgs in x_img_out:
            out_t, out_r = out_imgs[:, :3, ...], out_imgs[:, 3:, ...]
            # Pixel loss
            l_g_pix_t = self.cri_pix(out_t, target_t)
            pix_t_loss_list.append(l_g_pix_t)
            l_g_pix_r = self.cri_pix(out_r, target_r)
            pix_r_loss_list.append(l_g_pix_r)

            # Perceptual loss
            l_g_percep_t, _ = self.cri_perceptual(out_t, target_t)
            if l_g_percep_t is not None:
                per_loss_list.append(l_g_percep_t)

            # Gradient loss
            l_g_grad = self.cri_grad(out_t, target_t)
            grad_loss_list.append(l_g_grad)

        # Apply weights to losses
        l_g_pix_t = self.calculate_weighted_loss(pix_t_loss_list)
        l_g_pix_r = self.calculate_weighted_loss(pix_r_loss_list)
        l_g_percep_t = self.calculate_weighted_loss(per_loss_list)
        l_g_grad = self.calculate_weighted_loss(grad_loss_list)

        # Total loss
        loss_dict['l_g_pix_t'] = l_g_pix_t
        loss_dict['l_g_pix_r'] = l_g_pix_r
        loss_dict['l_g_percep_t'] = l_g_percep_t
        loss_dict['l_g_grad'] = l_g_grad
        l_g_total = l_g_pix_t + l_g_pix_r + l_g_percep_t + l_g_grad

        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, prog_bar=True, sync_dist=True)

        # Store outputs for visualization
        self.last_inp = inp
        self.last_output_clean = output_clean
        self.last_output_reflection = output_reflection
        self.last_target_t = target_t

        return l_g_total

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step.
        
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index for multiple validation sets.
            
        Returns:
            dict: Output dict with clean and reflection images.
        """
        # Validate batch has the expected fields
        required_keys = ['input']
        for key in required_keys:
            if key not in batch:
                rank_zero_warn(f"Required key '{key}' missing from batch during validation")
                return {'error': f"Missing required key: {key}"}

        # Save input image info
        inp = batch['input']

        # Handle missing inp_path gracefully
        if 'inp_path' in batch and len(batch['inp_path']) > 0:
            img_name = osp.splitext(osp.basename(batch['inp_path'][0]))[0]
        else:
            # Generate a fallback name if inp_path is missing
            img_name = f"sample_{batch_idx}"
            rank_zero_warn(f"'inp_path' key missing in batch, using fallback name: {img_name}")

        # Forward pass - uses EMA model if provided by the callback
        model = self.net_g
        with torch.no_grad():
            x_cls_out, x_img_out = model(inp)
            output_clean, output_reflection = x_img_out[-1][:, :3, ...], x_img_out[-1][:, 3:, ...]

        # Process images for metrics and visualization
        clean_img = tensor2img([output_clean])
        reflection_img = tensor2img([output_reflection])

        metric_data = {'img': clean_img}

        # Calculate validation loss if targets are available
        compute_loss = 'target_t' in batch and 'target_r' in batch and self.cri_pix is not None

        if compute_loss:
            target_t = batch['target_t']
            target_r = batch['target_r']

            # Calculate losses (only the final output)
            with torch.no_grad():
                # Pixel loss
                l_val_pix_t = self.cri_pix(output_clean, target_t) if self.cri_pix else 0
                l_val_pix_r = self.cri_pix(output_reflection, target_r) if self.cri_pix else 0

                # Perceptual loss
                l_val_percep_t = self.cri_perceptual(output_clean, target_t) if self.cri_perceptual else 0

                # Gradient loss
                l_val_grad = self.cri_grad(output_clean, target_t) if self.cri_grad else 0

                # Total validation loss
                val_loss = l_val_pix_t + l_val_pix_r
                if isinstance(l_val_percep_t, torch.Tensor):
                    val_loss += l_val_percep_t
                if isinstance(l_val_grad, torch.Tensor):
                    val_loss += l_val_grad

                # Log validation loss
                self.log('val/loss', val_loss, sync_dist=True)

                # Add individual loss components
                self.log('val/pix_t', l_val_pix_t, sync_dist=True)
                self.log('val/pix_r', l_val_pix_r, sync_dist=True)
                if isinstance(l_val_percep_t, torch.Tensor):
                    self.log('val/percep', l_val_percep_t, sync_dist=True)
                if isinstance(l_val_grad, torch.Tensor):
                    self.log('val/grad', l_val_grad, sync_dist=True)

            # Add target image for metrics
            target_t_img = tensor2img([target_t])
            metric_data['img2'] = target_t_img
        elif 'target_t' in batch:
            # Add target image for metrics even if loss computation is skipped
            target_t = batch['target_t']
            target_t_img = tensor2img([target_t])
            metric_data['img2'] = target_t_img
        else:
            # Log warning if ground truth is missing
            if self.trainer.is_global_zero and not batch.get('no_target_warning', False):
                rank_zero_warn("Ground truth images missing from validation batch, "
                               "metrics requiring ground truth will not be computed")

        # Save validation images
        if self.trainer.is_global_zero and self.opt['val'].get('save_img', False):
            try:
                if hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders:
                    try:
                        # 修复DataLoader不可下标访问的问题
                        if isinstance(self.trainer.val_dataloaders, list):
                            if len(self.trainer.val_dataloaders) > dataloader_idx:
                                if hasattr(self.trainer.val_dataloaders[dataloader_idx].dataset, 'opt'):
                                    dataset_name = self.trainer.val_dataloaders[dataloader_idx].dataset.opt['name']
                                else:
                                    dataset_name = f'val_{dataloader_idx}'
                            else:
                                dataset_name = f'val_{dataloader_idx}'
                        else:
                            # 如果val_dataloaders不是列表，可能只有一个dataloader
                            if hasattr(self.trainer.val_dataloaders.dataset, 'opt'):
                                dataset_name = self.trainer.val_dataloaders.dataset.opt['name']
                            else:
                                dataset_name = f'val_{dataloader_idx}'
                    except (AttributeError, IndexError, TypeError):
                        dataset_name = f'val_{dataloader_idx}'
                else:
                    dataset_name = f'val_{dataloader_idx}'

                current_iter = self.current_epoch if self.trainer.sanity_checking else self.global_step

                if self.opt['is_train']:
                    save_clean_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                   f'{img_name}_clean_{current_iter}.png')
                    save_reflection_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                        f'{img_name}_reflection_{current_iter}.png')
                else:
                    if self.opt['val'].get('suffix'):
                        save_clean_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                       f'{img_name}_clean_{self.opt["val"]["suffix"]}.png')
                        save_reflection_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                            f'{img_name}_reflection_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_clean_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                       f'{img_name}_clean_{self.opt["name"]}.png')
                        save_reflection_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                            f'{img_name}_reflection_{self.opt["name"]}.png')

                # Create directory if it doesn't exist
                os.makedirs(osp.dirname(save_clean_img_path), exist_ok=True)
                imwrite(clean_img, save_clean_img_path)
                imwrite(reflection_img, save_reflection_img_path)
            except Exception as e:
                # 使用rank_zero_warn代替self.logger.warning
                rank_zero_warn(f"Error saving validation images: {str(e)}")

        # Calculate metrics
        if 'img2' in metric_data and self.opt['val'].get('metrics') is not None:
            for name, opt_ in self.opt['val']['metrics'].items():
                try:
                    metric_value = calculate_metric(metric_data, opt_)
                    self.log(f'val/{name}', metric_value, sync_dist=True, add_dataloader_idx=False)
                    # Store for later aggregation
                    dataset_name = f'val_{dataloader_idx}'
                    if dataset_name not in self.current_val_metrics:
                        self.current_val_metrics[dataset_name] = {}
                    if name not in self.current_val_metrics[dataset_name]:
                        self.current_val_metrics[dataset_name][name] = []
                    self.current_val_metrics[dataset_name][name].append(metric_value)
                except Exception as e:
                    rank_zero_warn(f"Error calculating metric '{name}': {str(e)}")

        # Store for visualization
        if batch_idx == 0:
            self.val_inp = inp
            self.val_output_clean = output_clean
            self.val_output_reflection = output_reflection
            if 'target_t' in batch:
                self.val_target_t = batch['target_t']

        return {
            'output_clean': output_clean,
            'output_reflection': output_reflection,
            'img_name': img_name
        }

    def on_validation_epoch_start(self):
        """Setup metrics collection at the start of validation epoch."""
        self.current_val_metrics = {}

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and epoch.
        
        Args:
            dataset_name (str): Dataset name.
        """
        if not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        if dataset_name not in self.best_metric_results:
            record = dict()
            for metric, content in self.opt['val']['metrics'].items():
                better = content.get('better', 'higher')
                init_val = float('-inf') if better == 'higher' else float('inf')
                record[metric] = dict(better=better, val=init_val, epoch=-1)
            self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_epoch):
        """Update the best metric result.
        
        Args:
            dataset_name (str): Dataset name.
            metric (str): Metric name.
            val (float): Metric value.
            current_epoch (int): Current epoch.
        """
        if dataset_name not in self.best_metric_results:
            self._initialize_best_metric_results(dataset_name)

        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['epoch'] = current_epoch
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['epoch'] = current_epoch

    def on_validation_epoch_end(self):
        """Operations at the end of validation epoch."""
        # Calculate and log average metrics across all validation samples
        if self.current_val_metrics and self.trainer.is_global_zero:
            for dataset_name, metrics in self.current_val_metrics.items():
                log_str = f'Validation [{dataset_name}] Epoch {self.current_epoch}\n'

                self._initialize_best_metric_results(dataset_name)

                for metric_name, values in metrics.items():
                    avg_value = sum(values) / len(values)
                    log_str += f'\t # {metric_name}: {avg_value:.4f}'

                    # Update best metric
                    self._update_best_metric_result(dataset_name, metric_name, avg_value, self.current_epoch)

                    log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric_name]["val"]:.4f} @ '
                                f'epoch {self.best_metric_results[dataset_name][metric_name]["epoch"]}\n')

                    # Log to tensorboard
                    self.logger.experiment.add_scalar(
                        f'metrics/{dataset_name}/{metric_name}', avg_value, self.current_epoch
                    )

                # Log to console
                rank_zero_info(log_str)

    def test_step(self, batch, batch_idx):
        """Test step.
        
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            
        Returns:
            dict: Output dict with clean and reflection images.
        """
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            dict: Optimizer and scheduler configuration.
        """
        train_opt = self.opt['train']

        # Setup different parameter groups with their learning rates
        params_lr = [
            {'params': self.net_g.get_baseball_params(), 'lr': train_opt['optim_g']['baseball_lr']},
            {'params': self.net_g.get_other_params(), 'lr': train_opt['optim_g']['other_lr']},
        ]

        # Get optimizer configuration without modifying original config
        optim_type = train_opt['optim_g']['type']
        optim_config = {k: v for k, v in train_opt['optim_g'].items()
                        if k not in ['type', 'baseball_lr', 'other_lr']}

        # Create optimizer
        optimizer = self.get_optimizer(optim_type, params_lr, **optim_config)

        # Setup learning rate scheduler without modifying original config
        scheduler_type = train_opt['scheduler']['type']
        scheduler_config = {k: v for k, v in train_opt['scheduler'].items()
                            if k != 'type'}

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_type == 'CosineAnnealingRestartLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

        # Get the monitor metric from checkpoint config if available
        monitor_metric = self.opt.get('checkpoint', {}).get('monitor', 'val/psnr')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor_metric,  # Use the same metric as checkpoint monitor
                "interval": "epoch",
                "frequency": 1
            }
        }

    def get_optimizer(self, optim_type, params, **kwargs):
        """Get optimizer based on type.
        
        Args:
            optim_type (str): Optimizer type.
            params (list): Parameter groups.
            **kwargs: Additional optimizer arguments.
            
        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def calculate_weighted_loss(self, loss_list):
        """Calculate weighted loss.
        
        Args:
            loss_list (list): List of losses at different scales.
            
        Returns:
            torch.Tensor: Weighted loss.
        """
        if not loss_list:
            return 0
        weights = [0.25, 0.5, 0.75, 1.0]
        # Make sure we have enough weights
        while len(weights) < len(loss_list):
            weights.append(1.0)
        # Trim weights if we have too many
        weights = weights[:len(loss_list)]
        return sum(w * loss for w, loss in zip(weights, loss_list))

    def on_save_checkpoint(self, checkpoint):
        """Operations when saving checkpoint.
        
        Args:
            checkpoint (dict): Checkpoint dict.
        """
        # Save best metric results
        checkpoint['best_metric_results'] = self.best_metric_results
        # EMA model will be saved by EMACallback

    def on_load_checkpoint(self, checkpoint):
        """Operations when loading checkpoint.
        
        Args:
            checkpoint (dict): Checkpoint dict.
        """
        # Load best metric results
        if 'best_metric_results' in checkpoint:
            self.best_metric_results = checkpoint['best_metric_results']
        # EMA model will be loaded by EMACallback

    def get_current_visuals(self):
        """Get current visuals for visualization and comparison.
        
        Returns:
            OrderedDict: Dictionary of current visual tensors.
        """
        out_dict = OrderedDict()
        # Training visuals
        if hasattr(self, 'last_inp'):
            out_dict['inp'] = self.last_inp.detach().cpu()
        if hasattr(self, 'last_output_clean'):
            out_dict['result_clean'] = self.last_output_clean.detach().cpu()
        if hasattr(self, 'last_output_reflection'):
            out_dict['result_reflection'] = self.last_output_reflection.detach().cpu()
        if hasattr(self, 'last_target_t'):
            out_dict['target_t'] = self.last_target_t.detach().cpu()

        # If training visuals not available, try validation visuals
        if 'inp' not in out_dict and hasattr(self, 'val_inp'):
            out_dict['inp'] = self.val_inp.detach().cpu()
        if 'result_clean' not in out_dict and hasattr(self, 'val_output_clean'):
            out_dict['result_clean'] = self.val_output_clean.detach().cpu()
        if 'result_reflection' not in out_dict and hasattr(self, 'val_output_reflection'):
            out_dict['result_reflection'] = self.val_output_reflection.detach().cpu()
        if 'target_t' not in out_dict and hasattr(self, 'val_target_t'):
            out_dict['target_t'] = self.val_target_t.detach().cpu()

        return out_dict
