"""
Shape Distance Metrics for Trajectory Comparison

Implements DTW, Geometric (Point-to-Polyline), and Frechet distance
for comparing estimated trajectories to reference paths.

Translated from MATLAB digital-8 shape comparison code.
"""

import xp
from typing import Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def shape_distance(est_curve, ref: dict, dist_type: str = "DTW",ifplot: bool = True):
    """
    Calculate distance between estimated and reference curves
    
    Args:
        est_curve: Estimated trajectory (d x N or N x d)
        ref_curve: Reference trajectory (d x M or M x d)
        dist_type: Distance type - "DTW", "GEO", or "FRECHET"
        ifplot: Whether to plot results
        
    Returns:
        Distance metric value
    """

    ref_curve = xp.copy(ref["truth_sparse"])
    ref_dense = xp.copy(ref["truth"])
    if ref_curve is None and ref_dense is None:
        print("No reference, output is set to None")
        return None,0
    
    if ref_curve is None and ref_dense is not None:
        ref_curve = xp.copy(ref_dense)
    if ref_dense is None and ref_curve is not None:
        ref_dense = xp.copy(ref_curve)
    
    # Ensure curves are N x d (rows are points)
    if est_curve.shape[0] < est_curve.shape[1]:
        est_curve = est_curve.T
    if ref_curve.shape[0] < ref_curve.shape[1]:
        ref_curve = ref_curve.T
    if ref_dense.shape[0] < ref_dense.shape[1]:
        ref_dense = ref_dense.T

     
    N = est_curve.shape[0]
    
    dist_type = dist_type.upper()
    
    # Calculate distance based on type
    if dist_type == "DTW":
        # try:
        #     dist = compute_dtw_distance(est_curve, ref_dense, ifplot)
        # except Exception as e:
        #     print(f"DTW error: {e}")
        #     dist = xp.inf()
        #     if ifplot:
        #         plot_error_case(est_curve, ref_dense)

        dist, fig = compute_dtw_distance(est_curve, ref_dense, ifplot)
        # print(f"DTW distance: {dist:.20f}")

    elif dist_type == "PROJECT":
        dist, fig = compute_project_points_distance(est_curve, ref_curve, ifplot, W=10)

    elif dist_type == "FRECHET":
        dist, fig = frechet_distance(ref_curve, est_curve, ifplot)
        # print(f"Frechet distance: {dist:.20f}")
    
    else:
        raise ValueError(f"Unknown distance type: {dist_type}")
    
    return dist, fig

def dtw_distance(X, Y):
    """
    Dynamic Time Warping distance with warp path
    
    Args:
        X: First sequence (N x d)
        Y: Second sequence (M x d)
        
    Returns:
        Tuple of (distance, ix, iy) where ix, iy are warp path indices
    """
    N, M = X.shape[0], Y.shape[0]
    
    # Compute distance matrix
    D = xp.cdist(X, Y, metric='euclidean')
    
    # Initialize cost matrix
    C = xp.full((N + 1, M + 1), xp.inf())
    C[0, 0] = 0
    
    # Fill cost matrix
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            C[i, j] = D[i-1, j-1] + min(C[i-1, j], C[i, j-1], C[i-1, j-1])
    
    # Backtrack to find warp path
    i, j = N, M
    ix, iy = [i-1], [j-1]
    
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            # Choose minimum of three predecessors
            candidates = [C[i-1, j], C[i, j-1], C[i-1, j-1]]
            min_idx = xp.argmin(xp.asarray(candidates))
            
            if min_idx == 0:
                i -= 1
            elif min_idx == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        
        ix.append(i-1)
        iy.append(j-1)
    
    # Reverse to get forward path
    ix = xp.array(ix[::-1])
    iy = xp.array(iy[::-1])
    
    return C[N, M], ix, iy


def compute_dtw_distance(est_curve,ref_dense,ifplot: bool = True):
    """
    Compute DTW distance and shortest geometric distance
    
    Args:
        est_curve: Estimated curve (N x d)
        ref_dense: Dense reference curve (M x d)
        ifplot: Whether to plot
        
    Returns:
        Mean shortest DTW geometric distance
    """
    N = est_curve.shape[0]
    
    # Compute DTW
    dtw_dist, ix, iy = dtw_distance(est_curve, ref_dense)
    
    # Compute shortest geometric distance for each estimated point
    shortest_dtw_dist = xp.zeros(N)
    shortest_dtw_proj = xp.zeros((N, 2))
    
    for i in range(N):
        # Find which DTW pairs include est(i)
        idx = xp.as_int(xp.where(ix == i)[0])
        ref_idx = xp.as_int(iy[idx])
        
        # Compute distances to matched reference points
        d = xp.norm(est_curve[i, :] - ref_dense[ref_idx, :], axis=1)
        kmin = xp.argmin(d)
        
        shortest_dtw_dist[i] = d[kmin]
        shortest_dtw_proj[i, :] = ref_dense[ref_idx[kmin], :]
    
    mean_dist = xp.as_numpy(xp.mean(shortest_dtw_dist))
    
    if ifplot:
        ref_dense_np = xp.as_numpy(ref_dense)
        est_curve_np = xp.as_numpy(est_curve)
        shortest_dtw_proj_np = xp.as_numpy(shortest_dtw_proj)

        ix_np = xp.as_numpy(ix).astype(int)
        iy_np = xp.as_numpy(iy).astype(int)
        shortest_dtw_proj_np = xp.as_numpy(shortest_dtw_proj)
        shortest_dtw_dist_np = xp.as_numpy(shortest_dtw_dist)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'DTW Warping', 'DTW Distance Distribution'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}]]
        )

        fig.add_trace(
            go.Scatter(
                x=ref_dense_np[:, 0],
                y=ref_dense_np[:, 1],
                mode='lines',
                line=dict(color='black', width=2),
                name='True Path'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=est_curve_np[:, 0],
                y=est_curve_np[:, 1],
                mode='lines',
                line=dict(color='red', width=2),
                name='Estimated Path'
            ),
            row=1, col=1
        )

        if len(ix_np) > 0:

            dtw_lines_x = []
            dtw_lines_y = []
            for k in range(min(200, len(ix_np))):  
                dtw_lines_x.extend([est_curve_np[ix_np[k], 0], ref_dense_np[iy_np[k], 0], None])
                dtw_lines_y.extend([est_curve_np[ix_np[k], 1], ref_dense_np[iy_np[k], 1], None])

            fig.add_trace(
                go.Scatter(
                    x=dtw_lines_x,
                    y=dtw_lines_y,
                    mode='lines',
                    line=dict(color='gray', width=0.8),
                    opacity=0.3,
                    name='DTW Links',
                    showlegend=True
                ),
                row=1, col=1
            )

        shortest_lines_x = []
        shortest_lines_y = []
        for i in range(N):
            shortest_lines_x.extend([est_curve_np[i, 0], shortest_dtw_proj_np[i, 0], None])
            shortest_lines_y.extend([est_curve_np[i, 1], shortest_dtw_proj_np[i, 1], None])

        fig.add_trace(
            go.Scatter(
                x=shortest_lines_x,
                y=shortest_lines_y,
                mode='lines',
                line=dict(color='blue', width=1.5),
                opacity=0.5,
                name='Shortest DTW',
                showlegend=True
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(shortest_dtw_dist_np))),
                y=shortest_dtw_dist_np,
                mode='markers+lines',
                marker=dict(size=5, color='blue'),
                line=dict(width=1, color='lightblue'),
                name='Point Distance'
            ),
            row=1, col=2
        )

        fig.add_hline(
            y=mean_dist,
            line_dash="dash",
            line_color="red",
            line_width=2,
            opacity=0.7,
            annotation_text=f"Mean: {mean_dist:.4f}",
            annotation_position="top right",
            row=1, col=2
        )

        fig.update_layout(
            title=f"DTW Analysis - Accumulated: {xp.as_numpy(dtw_dist):.2f}",
            width=1400,
            height=600,
            template='plotly_white'
        )

        fig.update_xaxes(title_text='X', row=1, col=1, scaleanchor="y", scaleratio=1)
        fig.update_yaxes(title_text='Y', row=1, col=1)

        fig.update_xaxes(title_text='Point Index', row=1, col=2)
        fig.update_yaxes(title_text='Shortest DTW Distance', row=1, col=2)

        # fig.show()

    return mean_dist, fig

def compute_project_points_distance(est_traj, ref_traj, ifplot: bool = True, W=10, alpha=50.0):
    """
    Compute exact geometric perpendicular distance (point-to-polyline)
    
    Args:
        est_traj: Estimated curve (N x d)
        ref_traj: Reference curve (M x d)
        ifplot: Whether to plot
        
    Returns:
        Mean geometric distance
    """
    
    last_seg = 0
    results = []

    for p in est_traj:
        q, dist2, seg_soft, t = project_point_to_polyline_window(p, ref_traj, last_seg, W, alpha)
        results.append((p, q, dist2))
        if xp._xp.use_torch:
            last_seg = int(seg_soft.detach().round().clamp(0, ref_traj.shape[0]-2).item())
        else:
            last_seg = int(seg_soft)

    d2_list = [d2 for _,_,d2 in results]
    d2_stack = xp.stack(d2_list)
    mean_dist = xp.mean(d2_stack)
    max_dist = xp.max(d2_stack)

    
    if ifplot:
        ref_dense_np = xp.as_numpy(ref_traj)
        est_curve_np = xp.as_numpy(est_traj)
        proj_pts_np = xp.as_numpy([q for _,q,_ in results])
        dist2_all_np = xp.as_numpy([d2 for _,_,d2 in results])

        # fig = go.Figure()

        # fig.add_trace(
        #     go.Scatter(x=[i for i in range(len(dist2_all_np))], y=dist2_all_np, 
        #             mode='markers')
        # )
        # fig.show()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=ref_dense_np[:, 0], y=ref_dense_np[:, 1], 
                    mode='lines', name='True Path', line=dict(color='black', width=1))
        )
        fig.add_trace(
            go.Scatter(x=est_curve_np[:, 0], y=est_curve_np[:, 1], 
                    mode='lines', name='Estimated Path')
        )

        perp_x = []
        perp_y = []
        for i in range(est_traj.shape[0]):  
            perp_x.extend([est_curve_np[i, 0], proj_pts_np[i, 0], None])
            perp_y.extend([est_curve_np[i, 1], proj_pts_np[i, 1], None])

        fig.add_trace(
            go.Scatter(x=perp_x, y=perp_y, mode='lines',
                    line=dict(color='gray', width=1),
                    name='Perpendiculars', showlegend=True)
        )

        fig.update_layout(
            title=f'Perpendicular Distances (mean: {mean_dist:.20f}, max: {max_dist:.20f})',
            xaxis=dict(scaleanchor="y", scaleratio=1),
            width=800,
            height=600
        )

        # fig.show()


    return max_dist, fig

def project_point_to_polyline_window(p, polyline, last_seg=0, W=10, alpha=50):
    M = len(polyline)
    start = max(0, last_seg - W)
    end = min(M - 1, last_seg + W)

    qs = []
    d2s = []
    ts = []
    seg_ids = []

    for i in range(start, end):
        q, t = project_point_to_segment(p, polyline[i], polyline[i+1])
        d2 = (p - q) @ xp.transpose(p - q)

        qs.append(q)
        d2s.append(d2)
        ts.append(t)
        seg_ids.append(i)

    qs = xp.stack(qs)       # (W,d)
    d2s = xp.stack(d2s)     # (W,)
    ts =  xp.stack(ts)      # (W,)

    w = xp.softmax(-alpha * d2s, axis=0)

    q_soft = (w[:,None] * qs).sum(dim=0)
    d_soft = (w * d2s).sum()
    t_soft = (w * ts).sum()

    seg_ids = xp.array(seg_ids)
    seg_soft = (w * seg_ids).sum()

    return q_soft, d_soft, seg_soft, t_soft

        # mask = d2 < min_dist2
        # min_dist2 = xp.where(mask, d2, min_dist2)
        # print(1)
        # best_q    = xp.where(mask.unsqueeze(-1), q, best_q)
        # print(2)
        # best_t    = xp.where(mask, t, best_t)
        # print(3)
        # best_seg  = xp.where(mask, xp.array(i), best_seg)
        # print(4)
    # th = xp.array(fallback_thresh)
    # mask = min_dist2 > th
    

    # q_glb, d2_glb, seg_glb, t_glb = project_point_to_polyline_global(p, polyline)

    # best_q = xp.where(mask.unsqueeze(-1), q_glb, best_q)
    # min_dist2 = xp.where(mask, d2_glb, min_dist2)
    # best_t = xp.where(mask, t_glb, best_t)


    # return best_q, min_dist2, best_seg, best_t

def project_point_to_segment(p, a, b):
    """
    Args:
        p: a estimated point
        a, b: vertices of a segment line

    Return:
        q: projected point
        t: direction of the segment
    """
    ab = b - a
    ap = p - a
    denom = xp.dot(ab, ab)
    denom_safe = denom + 1e-12
    t = xp.dot(ap, ab) / denom_safe

    t = xp.clip(t, 0.0, 1.0)
    q = a + t * ab

    return q, t


# def project_point_to_polyline_global(p, polyline):
#     min_dist2 = xp.inf()
#     best_q = None
#     best_seg = 0
#     best_t = 0.0

#     for i in range(len(polyline)-1):
#         q, t = project_point_to_segment(p, polyline[i], polyline[i+1])
#         d2 = (p - q) @ xp.transpose(p - q)

#         mask = d2 < min_dist2

#         min_dist2 = xp.where(mask, d2, min_dist2)
#         best_q    = xp.where(mask.unsqueeze(-1), q, best_q)
#         best_t    = xp.where(mask, t, best_t)
#         best_seg  = xp.where(mask, xp.array(i), best_seg)


#     return best_q, min_dist2, best_seg, best_t



def frechet_distance(P, Q, ifplot: bool=True):
    """
    Discrete Frechet distance (recursive with memoization)
    
    Args:
        P: First curve (N x d)
        Q: Second curve (M x d)
        
    Returns:
        Frechet distance
    """
    
    def recurse(i: int, j: int):
        """Recursive helper with memoization"""
        if ca[i, j] > -0.5:
            return ca[i, j]
        
        if i == 0 and j == 0:
            val = xp.norm(P[0, :] - Q[0, :])
        elif i == 0:
            val = max(recurse(0, j - 1), xp.norm(P[0, :] - Q[j, :]))
        elif j == 0:
            val = max(recurse(i - 1, 0), xp.norm(P[i, :] - Q[0, :]))
        else:
            val = max(
                min([
                    recurse(i - 1, j),
                    recurse(i - 1, j - 1),
                    recurse(i, j - 1)
                ]),
                xp.norm(P[i, :] - Q[j, :])
            )
        
        ca[i, j] = val
        return val
    
    n, m = P.shape[0], Q.shape[0]
    ca = -xp.ones((n, m))

    dist = recurse(n - 1, m - 1)

    if ifplot:
        """Plot only the curves without distance lines"""
        ref_dense_np = xp.as_numpy(P)
        est_curve_np = xp.as_numpy(Q)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=ref_dense_np[:, 0],
                y=ref_dense_np[:, 1],
                mode='lines+markers',
                line=dict(color='black', width=1),
                marker=dict(size=3, color='black'),
                name='True Path',
                hovertemplate='True<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=est_curve_np[:, 0],
                y=est_curve_np[:, 1],
                mode='lines',
                line=dict(color='red', width=2),
                name='Estimated Path',
                hovertemplate='Estimate<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            )
        )

        fig.update_layout(
            title=dict(
                text=f'Trajectory (Fréchet Distance) {dist:.20f}',
                font=dict(size=18)
            ),
            xaxis_title='X',
            yaxis_title='Y',
            template='plotly_white',
            width=800,
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        fig.update_xaxes(
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        )

        # fig.show()

    return dist, fig


def plot_error_case(est_curve, ref_dense):
    """Plot error case with no distance lines"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xp.as_numpy(ref_dense[:, 0]),
            y=xp.as_numpy(ref_dense[:, 1]),
            mode='lines',
            line=dict(color='black', width=1),
            name='True Path'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xp.as_numpy(est_curve[:, 0]),
            y=xp.as_numpy(est_curve[:, 1]),
            mode='lines',
            line=dict(color='red', width=1),
            name='Estimated Path'
        )
    )

    fig.update_layout(
        title='Error: DTW Failed',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=800,
        height=600
    )

    fig.show()
    

# Example usage
if __name__ == "__main__":
    print("Testing shape distance metrics...")
    
    # Create a simple reference curve (figure-8)
    t = xp.linspace(0, 2*xp.pi, 100)
    ref_x = xp.sin(t)
    ref_y = xp.sin(2*t)
    ref_curve = xp.column_stack([ref_x, ref_y])
    
    # Create noisy estimated curve
    noise = 0.05 * xp.randn(*ref_curve.shape)
    est_curve = ref_curve + noise
    
    print("\n=== DTW Distance ===")
    dtw_dist = shape_distance(est_curve, ref_curve, dist_type="DTW", ifplot=True)
    print(f"DTW distance: {dtw_dist:.4f}")
    
    print("\n=== Geometric Distance ===")
    geo_dist = shape_distance(est_curve, ref_curve, dist_type="GEO", ifplot=True)
    print(f"Geometric distance: {geo_dist:.4f}")
    
    print("\n=== Frechet Distance ===")
    frechet_dist = shape_distance(est_curve, ref_curve, dist_type="FRECHET", ifplot=False)
    print(f"Frechet distance: {frechet_dist:.4f}")
    
    print("\n✓ All distance metrics computed successfully!")