from __future__ import division, print_function

import torch
import numpy as np
import time

from package.utils.data_utils import batch_diag


def evolve_active_cvnet(alpha, beta, init_contour, init_contour0, delta_t=1e-2, max_steps=128):

    assert (alpha.device == beta.device == init_contour.device)
    batch_size, height, width = alpha.size()
    _, L, _ = init_contour.size()

    init_contour = init_contour.clone()
    init_contour0 = init_contour0.clone()

    contour_x = init_contour[..., 0] # [N, L]
    contour_y = init_contour[..., 1] # [N, L]
    contour_x0 = init_contour0[..., 0]
    contour_y0 = init_contour0[..., 1]

    batch_select_idx = torch.arange(batch_size).unsqueeze(1).repeat(1, L).flatten()

    for i in range(max_steps):
        exceeded_left = contour_x < 0
        exceeded_right = contour_x > (width - 1)
        exceeded_top = contour_y < 0
        exceeded_bottom = contour_y > (height - 1)

        x1 = contour_x.floor().float()
        y1 = contour_y.floor().float()

        x1 = torch.max(x1, torch.ones_like(x1))
        y1 = torch.max(y1, torch.ones_like(y1))        
        x1 = torch.min(x1, torch.ones_like(x1) * (width - 2))
        y1 = torch.min(y1, torch.ones_like(y1) * (height - 2))

        x2 = (x1 + 1).float()
        y2 = (y1 + 1).float()
        x1 = x1 - 1
        y1 = y1 - 1

        x1_idx = x1.long().flatten()
        y1_idx = y1.long().flatten()
        x2_idx = x2.long().flatten()
        y2_idx = y2.long().flatten()

        alpha_Q11 = alpha[batch_select_idx, y1_idx, x1_idx].view(batch_size, -1)
        alpha_Q12 = alpha[batch_select_idx, y2_idx, x1_idx].view(batch_size, -1)
        alpha_Q21 = alpha[batch_select_idx, y1_idx, x2_idx].view(batch_size, -1)
        alpha_Q22 = alpha[batch_select_idx, y2_idx, x2_idx].view(batch_size, -1)
        alpha_i = (alpha_Q11 * (x2 - contour_x) * (y2 - contour_y) +
                  alpha_Q21 * (contour_x - x1) * (y2 - contour_y) +
                  alpha_Q12 * (x2 - contour_x) * (contour_y - y1) + 
                  alpha_Q22 * (contour_x - x1) * (contour_y - y1)) / ((x2 - x1) * (y2 - y1))


        beta_Q11 = beta[batch_select_idx, y1_idx, x1_idx].view(batch_size, -1)
        beta_Q12 = beta[batch_select_idx, y2_idx, x1_idx].view(batch_size, -1)
        beta_Q21 = beta[batch_select_idx, y1_idx, x2_idx].view(batch_size, -1)
        beta_Q22 = beta[batch_select_idx, y2_idx, x2_idx].view(batch_size, -1)
        beta_i = (beta_Q11 * (x2 - contour_x) * (y2 - contour_y) +
                  beta_Q21 * (contour_x - x1) * (y2 - contour_y) +
                  beta_Q12 * (x2 - contour_x) * (contour_y - y1) + 
                  beta_Q22 * (contour_x - x1) * (contour_y - y1)) / ((x2 - x1) * (y2 - y1))

        alpha_i = alpha_i.clone()
        beta_i = beta_i.clone()
        alpha_i[exceeded_left | exceeded_right | exceeded_top | exceeded_bottom] = 0
        beta_i[exceeded_left | exceeded_right | exceeded_top | exceeded_bottom] = 0

        a = -2 * alpha_i
        b = alpha_i[:, 1:L]
        b_corner = alpha_i[:,0].unsqueeze(1)
        c = alpha_i[:, :L-1]
        c_corner = alpha_i[:,L-1].unsqueeze(1)

        A = (
            batch_diag(a) + batch_diag(b, diagonal=1) + batch_diag(c, diagonal=-1) + 
            batch_diag(b_corner, diagonal=-(L-1)) + batch_diag(c_corner, diagonal=(L-1))
        )
        delta_x = torch.matmul(A, contour_x.unsqueeze(2)).squeeze(2) * delta_t ** 2 - beta_i * (contour_x - contour_x0) * delta_t
        delta_y = torch.matmul(A, contour_y.unsqueeze(2)).squeeze(2) * delta_t ** 2 - beta_i * (contour_y - contour_y0) * delta_t
        new_x = 2 * contour_x - contour_x0 + delta_x
        new_y = 2 * contour_y - contour_y0 + delta_y

        contour_x0 = contour_x
        contour_y0 = contour_y
        contour_x = new_x
        contour_y = new_y

    return contour_x, contour_y





