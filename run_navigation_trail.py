import xp
import numpy as np
from ssm_hybvio import SSM_HybVIO
from ssm_sola import SSM_Sola
from ssm_openshoe import SSM_OpenShoe
from ssm_matern import SSM_Matern
from navigation_loop import navigation_loop
# from params_optimization import ParamsOptimizer
from prepare_my_data import load_imu_data, load_truth_data, plot_truth, GlobalConfig
from types import SimpleNamespace

def run_navigation_trail(config: GlobalConfig, nav_flag: bool = True):
    """
    Run a single navigation trail
    
    Args:
        trail: Trial number
        nav_flag: If True, run navigation; if False, run optimization
    """
    # ========== Load Data ==========
    imu_raw, imu_file = load_imu_data(config)
    truth, truth_sparse = load_truth_data(config, 1/imu_raw.Ts)

    if config.plot:
        imu_raw.plot_imu(title="Loaded IMU data")
        # plot_truth(truth,truth_sparse,axis2D=config.axis2D)

    still_time = config.still_time

    # imu.detect_still_segments(window_sec=0.1,
    #                         std_threshold=0.005,
    #                         mean_threshold=0.02)

    # ========== Setup Navigation ==========

    # Initialize SINS object
    ini_yaw = 0
    initial_pos = xp.zeros((3,1))
    if config.SSM_name.upper() == "HYBVIO":
        _hkw = config.hybvio_kwargs or {}
        my_sins = SSM_HybVIO(Ts=imu_raw.Ts,init_pos=initial_pos,imu_uncal=imu_raw,init_yaw=ini_yaw,still=still_time,**_hkw)
        my_sins.estimator_type = "EKF"
    elif config.SSM_name.upper() == "SOLA":
        _skw = config.sola_kwargs or {}
        my_sins = SSM_Sola(Ts=imu_raw.Ts,init_pos=initial_pos,imu_uncal=imu_raw,init_yaw=ini_yaw,still=still_time,**_skw)
        my_sins.estimator_type = "ErKF"
    elif config.SSM_name.upper() == "OPENSHOE":
        _okw = config.openshoe_kwargs or {}
        my_sins = SSM_OpenShoe(Ts=imu_raw.Ts,init_pos=initial_pos,imu_uncal=imu_raw,init_yaw=ini_yaw,still=still_time,biases_switch='on',scalefactors_switch='off',**_okw)
        my_sins.estimator_type = "ErKF"
    elif config.SSM_name.upper() == "MATERN":
        _mkw = config.matern_kwargs or {}
        my_sins = SSM_Matern(Ts=imu_raw.Ts,init_pos=initial_pos,imu_uncal=imu_raw,init_yaw=ini_yaw,still=still_time,**_mkw)
        my_sins.estimator_type = "ErKF"
    else:
        print("Please specify valid SSM name: ""HYBVIO"", ""SOLA"", ""MYESKF"", ""OPENSHOE"", or ""MATERN""")

    my_sins.metric_type = config.metric_type
    my_sins.nav_mode = config.nav_mode
    my_sins.set_measurement_fixed_part('zupt')
    my_sins.R0 = xp.copy(my_sins.R)  

    # Set ground truth
    my_sins.truth = xp.copy(truth)
    my_sins.truth_sparse = xp.copy(truth_sparse)
    
    # # ========== ZUPT Detection ==========
    my_sins.ZVDtype = config.ZVDtype
    my_sins.min_zupt_duration_s = float(config.min_zupt_duration_s)
    my_sins.use_reference_zupt = bool(config.use_reference_zupt)
    my_sins.zupt_threshold = config.zupt_threshold
    
    
    # # Plot ZUPT detection
    # my_sins.plot_zupt_detection(zupt_all, u = {'Acc norm':xp.norm(u_all[0:3, :], axis=0)})

    # # Extract ROI
    # u_roi = u_all[:, roi]
    # zupt_roi = zupt_all[roi]
    # data_length = u_roi.shape[1]
    # my_sins.zupt_all = zupt_roi
    # my_sins.u_all = u_roi
    
    # print(f"Data length: {data_length} samples")
    
    # if "zupt_ref_external" in config and config.zupt_ref_external is not None:
    #     zupt_ref_ext = _resample_binary_to_length(config.zupt_ref_external, my_sins.u_all.shape[1])
    #     my_sins.zupt_ref = xp.asarray(zupt_ref_ext)
    if imu_raw.zupt_ref is not None:
        # print("Do you have special rule for converting the zupt_ref?")
        my_sins.zupt_ref = xp.copy(imu_raw.zupt_ref)

    # ========== Parameter Optimization Mode ==========
    if not nav_flag:
        pass
       
    
    # ========== Navigation Mode ==========
    else:
        # Dispatch to the appropriate navigation loop
        loss, my_sins, my_estimator, my_results = navigation_loop(
            my_sins, params=config.params_user,
            use_external_zupt=bool(config.use_reference_zupt)
        )

        out_main = SimpleNamespace(
                                    my_sins=my_sins,
                                    my_estimator=my_estimator,
                                    my_results=my_results
                                )
        if bool(config.use_reference_zupt): # also run with traditional ZVD for comparison
            loss, my_sins2, my_estimator2, my_results2 = navigation_loop(
                my_sins, params=config.params_user, use_external_zupt=False
            )
            out_more = SimpleNamespace(
                                        my_sins=my_sins2,
                                        my_estimator=my_estimator2,
                                        my_results=my_results2
                                    )
        else:
            out_more = SimpleNamespace(
                                        my_sins=None,
                                        my_estimator=None,
                                        my_results=None
                                    )

        return out_main, out_more

if __name__ == "__main__":
    pass