# Original work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>
# Modified work Copyright 2024 VUNO Inc. <minje.park@vuno.co>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable

import torch


def get_optimizer_from_config(
    config: dict,
    param_groups: Iterable[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    opt_name = config['optimizer']
    lr = config['lr']
    weight_decay = config['weight_decay']
    kwargs = config.get('optimizer_kwargs', {})
    if opt_name == "sgd":
        opt_cls = torch.optim.SGD
        kwargs['momentum'] = kwargs.get('momentum', 0)
    elif opt_name == "adamw":
        opt_cls = torch.optim.AdamW
        betas = kwargs.get('betas', (0.9, 0.999))
        if isinstance(betas, list):
            betas = tuple(betas)
        kwargs['betas'] = betas
        kwargs['eps'] = kwargs.get('eps', 1e-8)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    optimizer = opt_cls(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs,
    )
    return optimizer
