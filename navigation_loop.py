"""
EKF Navigation Loop with Smoothing

Main navigation loop for Extended Kalman Filter (EKF) or Error-State KF
with optional smoothing capability.
"""


# import numpy as np
import xp
from generic_filter_smoother import GenericFilterSmoother
from shape_distance import shape_distance
from tqdm import tqdm
show_bar = False

def navigation_loop(obj_sins, params = None, use_external_zupt: bool = False):
    """
    EKF navigation loop with smoothing
    
    Args:
        estimator_type: Type of estimator ("EKF", "ErKF", "ERTS", "ErRTS")
        obj_sins: SINS object (navigation system)
        u_all: IMU data (6 x N array), rows [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        zupt_all: Zero velocity update flags (N array)
        log_params: Optional parameters for optimization
        
    Returns:
        If log_params provided:
            xp.ndarray: [NLL, pos_error, quat_error]
        Else:
            Tuple: (obj_sins, estimator, results)
    """
    # save_dir = 'results'
    # os.makedirs(save_dir, exist_ok=True)

    # --------- Reinitialize the SINS object ---------
    obj_sins.re_initialize()


    # --------- Parameter handling (for optimization) ---------
    if params is not None:
        # DO SOMETHING
        pass

    else:
        pass

    # ========== ZUPT Detection ==========
    if not obj_sins.nav_mode: # in params optimization mode, use external zupt based on obj_sins.use_reference_zupt
        use_external_zupt = obj_sins.use_reference_zupt
    else: # in nav mode, use external zupt based on the input use_external_zupt
        pass

    if use_external_zupt:
        if hasattr(obj_sins, "zupt_ref") and obj_sins.zupt_ref is not None:
            zupt_all = xp.copy(obj_sins.zupt_ref)
            zupt_T = None
            print("Using ZUPT reference")
        else:
            raise ValueError("use_external_zupt is set as True but no valid zupt_ref in obj_sins is given.")
    else:
        zupt_all, zupt_T = obj_sins.zero_velocity_detector(obj_sins.u_all)
    
    obj_sins.zupt_all = zupt_all
    obj_sins.zupt_T = zupt_T

    # -------- prepare results container ----------
    data_length = obj_sins.u_all.shape[1]
    results = obj_sins.results_container(data_length)

    # -------- create estimator instant ----------
    estimator = GenericFilterSmoother(obj_sins.estimator_type)
    nll = xp.array([0])

    # --------- Interval tracking for position at ZVI updates ---------
    current_interval_idx = 0
    in_interval = False
    last_zupt_index = 0
    d = xp.array([0.0])
    # --------- Main navigation loop ----------
    # print("Running filtering loop...\n")
    for k in tqdm(range(data_length), disable=not show_bar):
        # Set IMU measurements
        obj_sins.u = {
            'acc': xp.reshape(obj_sins.u_all[0:3, k], (3,1)),
            'gyro': xp.reshape(obj_sins.u_all[3:6, k], (3,1))
        }
        
        # Prediction step
        estimator.predict(obj_sins)

        obj_sins.zvd_k = xp.copy(obj_sins.zupt_all[k])
        # print("current zvd_k:", obj_sins.zvd_k)
        # Measurement update step 
        if obj_sins.zupt_all[k] == 1:
            # Entering a new stationary interval
            if not in_interval:
                current_interval_idx += 1
                in_interval = True
            
            # Zero velocity update (ZUPT)
            obj_sins.measureType = 'ZUPT'
            obj_sins.measurement = xp.zeros((3,1))
            estimator.update(obj_sins)
            obj_sins.normalize_quaternions()
            nll += estimator.nll_k()

        else:

            in_interval = False
                
        
        # Optional additional updates (commented out)
        # Zero height update
        # if zero_height_upt[k] == 1:
        #     obj_sins.measureType = 'ZeroHeight'
        #     obj_sins.measurement = xp.array([0])
        #     estimator.update(obj_sins)
        #     obj_sins.normalize_quaternions()
        #     nll += estimator.nll_k()
        
        # Orientation update
        # if orientation_upt[k] == 1:
        #     obj_sins.measureType = 'Orientation'
        #     obj_sins.measurement = obj_sins.qua0
        #     estimator.update(obj_sins)
        #     obj_sins.normalize_quaternions()
        #     nll += estimator.nll_k()
        
        # Zero rotation update (ZRUPT)
        # if zrupt[k] == 1:
        #     obj_sins.measureType = 'ZRUPT'
        #     obj_sins.measurement = xp.zeros(3)
        #     estimator.update(obj_sins)
        #     obj_sins.normalize_quaternions()
        
        # Position X update
        # if pos_upt[k] == 1:
        #     obj_sins.measureType = 'PositionX'
        #     obj_sins.measurement = xp.array([0])
        #     estimator.update(obj_sins)
        #     obj_sins.normalize_quaternions()
        #     nll += estimator.nll_k()
        
        # Pseudo horizontal speed update
        # if pseudo_vel_upt[k] == 1:
        #     obj_sins.measureType = 'PseudoHoriSpeed'
        #     obj_sins.measurement = xp.array([default_speed])
        #     estimator.update(obj_sins)
        #     obj_sins.normalize_quaternions()
        

        # Store results
        obj_sins.store_result(results, k)
        estimator.nll_sum = nll
        
    # ====== Smoother ======
    results = estimator.smoother(obj_sins, results)
  
    if obj_sins.metric_type.upper() in ['FRECHET', 'DTW', 'GEO', 'PROJECT']:
        # ======= shape distance =========
        # shape_dist = shape_distance(results['ms'][0:2,:],{'truth': obj_sins.truth[0:2,:],'truth_sparse': obj_sins.truth_sparse[0:2,:]},obj_sins.metric_type,ifplot=False)
        # # plt.suptitle("smoothing")
        if obj_sins.truth is not None:
            truth = obj_sins.truth[0:2,:]
        else:
            truth = None
        if obj_sins.truth_sparse is not None:
            truth_sparse = obj_sins.truth_sparse[0:2,:]
        else:
            truth_sparse = None
        shape_dist, fig_shape_dist = shape_distance(results['POS'][0:2,:],{'truth': truth,'truth_sparse': truth_sparse},obj_sins.metric_type,ifplot=True)
#         fig.update_layout(
#     title=dict(
#         text="<b>Smoothing Analysis</b>",
#         font=dict(size=24, family='Arial Black'),
#         x=0.5,
#         y=0.98
#     )
# )
        # print("Shape distance calculated.\n")

        if obj_sins.nav_mode:
            fig_shape_dist.show()
        else:
            pass


        loss =  xp.copy(shape_dist)

    elif obj_sins.metric_type.upper() == 'D':
        loss =  d / obj_sins.truth_sparse.shape[1] 

    elif obj_sins.metric_type.upper() == 'DIST':
        pos_start = results['POS'][0:2, 0]
        pos_end = results['POS'][0:2, -1]
        loss = abs(xp.sqrt(xp.sum((pos_end - pos_start)**2)) - 0.198)

    elif obj_sins.metric_type.upper() == 'ORI':
        if 'ORI' in results:
            loss = xp.mean(xp.sum((results['ORI'] - obj_sins.qua0.reshape(-1, 1))**2, axis=0))
        elif 'ATT' in results:
            loss = xp.mean(xp.sum((results['ATT'] - obj_sins.att0.reshape(-1, 1))**2, axis=0))

    elif obj_sins.metric_type.upper() == 'NLL':
        loss =  xp.copy(estimator.nll_sum)

    else:
        print("Unknown metric type. Using NLL as default.")
        loss =  xp.copy(estimator.nll_sum)
        

    # filename = f'{obj_sins.metric_type}_{params}_{loss}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    # fig.write_image(os.path.join(save_dir, f"{filename}.png"), 
    #                 scale=3) 
    # plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')

    
    return loss, obj_sins, estimator, results


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the navigation loop
    """
    # This is just a template - actual usage requires proper initialization
    
    from ssm_hybvio import SSM_HybVIO
    from ssm_openshoe import SSM_OpenShoe
    from ssm_sola import SSM_Sola
    
    # Example parameters
    Ts = 0.01  # 100 Hz
    init_pos = xp.array([0, 0, 0])
    init_yaw = 0
    
    # Load or generate IMU data
    u_all = xp.random.randn(6, 1000)  # Example IMU data
    zupt_all = xp.zeros(1000)  # Example ZUPT flags
    zupt_all[100:200] = 1  # Mark some intervals as stationary
    
    # Create SINS object (choose one)
    # obj_sins = SSM_HybVIO(Ts, init_pos, u_all[0:3, 0:10], init_yaw)
    # obj_sins = SSM_OpenShoe(Ts, init_pos, u_all[0:3, 0], init_yaw)
    obj_sins = SSM_Sola(Ts, init_pos, u_all[0:3, 0:10], init_yaw)
    
    # Run navigation loop with EKF
    obj_sins_out, estimator, results = navigation_loop(
        obj_sins=obj_sins,
    )
    
    # Plot results
    obj_sins_out.plot_results(results, 'traj', 'att')
    
    print("EKF Navigation Loop module loaded successfully")
    print("Use navigation_loop() to run the navigation filter")
