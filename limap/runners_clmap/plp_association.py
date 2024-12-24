import os
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.vplib as _vplib
import limap.structures as _structure
import _limap._clmap as _clmap


def plp_association(cfg, imagecols, all_2d_lines, vpresults, pointtracks, linetracks, all_bpt2ds, vptracks, all_bpt2ds_vp, inf_planes3d, pp_bpt3d, lp_bpt3d):
    cfg_copy = copy.deepcopy(cfg)

    if cfg_copy["use_adaptive_lw"]:
        plp_associator_config = _clmap.PLPAssociatorConfig(cfg_copy)
        plp_associator = _clmap.PLPAssociator(plp_associator_config)
        plp_associator.InitImagecols(imagecols)
        plp_associator.InitPointTracks(pointtracks)
        plp_associator.InitLineTracks(linetracks)
        plp_associator.Init2DBipartites_PointLine(all_bpt2ds)
        plp_associator.InitVPTracks(vptracks)
        plp_associator.Init2DBipartites_VPLine(all_bpt2ds_vp)
        plp_associator.InitPlanes(inf_planes3d)
        plp_associator.InitPP_Bipartite3d(pp_bpt3d)
        plp_associator.InitLP_Bipartite3d(lp_bpt3d)

        plp_associator.ComputePointsUncertainty(cfg_copy['point_u_method'])
        plp_associator.ComputePointsOnLineUncertainty(cfg_copy['line_u_method'])

        print("Computing adaptive loss weights...", flush=True)
        plp_associator.ComputeAdaptiveLossWeight()
        print("print adaptive weights: ")
        plp_associator.PrintLossWeight()
        all_loss_weight = plp_associator.GetAllLossWeight()
        print("all loss weight =", all_loss_weight)
        cfg_copy["lw_point"] = all_loss_weight["lw_point"]
        cfg_copy["lw_pointline_association"] = all_loss_weight["lw_pointline_association"]
        cfg_copy["lw_vpline_association"] = all_loss_weight["lw_vpline_association"]
        cfg_copy["lw_vp_orthogonality"] = all_loss_weight["lw_vp_orthogonality"]
        cfg_copy["lw_vp_collinearity"] = all_loss_weight["lw_vp_collinearity"]
        cfg_copy["lw_pointplane_association"] = all_loss_weight["lw_pointplane_association"]
        cfg_copy["lw_lineplane_distance_association"] = all_loss_weight["lw_lineplane_distance_association"]
        cfg_copy["lw_lineplane_angle_association"] = all_loss_weight["lw_lineplane_angle_association"]
        cfg_copy["lw_plane_orthogonality"] = all_loss_weight["lw_plane_orthogonality"]
        cfg_copy["lw_plane_parallelism"] = all_loss_weight["lw_plane_parallelism"]

    # restart
    print("PLPAssociatorConfig =", cfg_copy)
    plp_associator_config = _clmap.PLPAssociatorConfig(cfg_copy)
    plp_associator = _clmap.PLPAssociator(plp_associator_config)
    plp_associator.InitImagecols(imagecols)
    plp_associator.InitPointTracks(pointtracks)
    plp_associator.InitLineTracks(linetracks)
    plp_associator.Init2DBipartites_PointLine(all_bpt2ds)
    plp_associator.InitVPTracks(vptracks)
    plp_associator.Init2DBipartites_VPLine(all_bpt2ds_vp)
    plp_associator.InitPlanes(inf_planes3d)
    plp_associator.InitPP_Bipartite3d(pp_bpt3d)
    plp_associator.InitLP_Bipartite3d(lp_bpt3d)

    plp_associator.ComputePointsUncertainty(cfg_copy['point_u_method'])
    plp_associator.ComputePointsOnLineUncertainty(cfg_copy['line_u_method'])

    plp_associator.UpdatePL_Bipartite3d(1e8)
    plp_associator.UpdateVPLine_Bipartite3d(1e8)
    pl_bpt3d_before_ba = plp_associator.GetPL_Bipartite3d()
    vpline_bpt3d_before_ba = plp_associator.GetVPLine_Bipartite3d()

    print("Setup...", flush=True)
    plp_associator.SetUp()
    print("Optimizing...", flush=True)
    plp_associator.Solve()

    plp_associator.UpdatePointTracks()
    plp_associator.UpdateLineTracks(cfg_copy['num_outliers_aggregate'])
    plp_associator.UpdateUncertainty(cfg_copy['point_u_method'], cfg_copy['line_u_method'])
    plp_associator.UpdatePL_Bipartite3d(cfg_copy['th_hard_pl_si_dist3d'])
    plp_associator.UpdateVPLine_Bipartite3d(cfg_copy['th_hard_vpline_angle3d'])
    plp_associator.UpdatePlanes()
    plp_associator.UpdatePP_Bipartite3d()
    plp_associator.UpdateLP_Bipartite3d()

    # iteratively optimization
    if (cfg_copy['use_vpline'] and cfg_copy['merge_vp']):
        n_iters = 0
        max_iter_num = 5
        while n_iters <= max_iter_num:
            n_iters += 1

            # update vps
            vptracks_opt_map = plp_associator.GetOutputVPTracks()
            vptracks_opt = [vptrack for (idx, vptrack) in vptracks_opt_map.items()]

            # vp3d id has been changed
            vptracks_opt_merged = _vplib.MergeVPTracksByDirection(vptracks_opt, 1.0)
            if len(vptracks_opt_merged) == len(vptracks_opt):
                break
            # run optimization on the merged vptracks
            all_bpt2ds_vp_opt = _structure.GetAllBipartites_VPLine2d(all_2d_lines, vpresults, vptracks_opt_merged)
            plp_associator.InitVPTracks(vptracks_opt_merged)
            plp_associator.Init2DBipartites_VPLine(all_bpt2ds_vp_opt)

            print("re-optim iter =", n_iters)

            # optimize again
            print("Setup...", flush=True)
            plp_associator.SetUp()
            print("Optimizing...", flush=True)
            plp_associator.Solve()

            plp_associator.UpdatePointTracks()
            plp_associator.UpdateLineTracks(cfg_copy['num_outliers_aggregate'])
            plp_associator.UpdateUncertainty(cfg_copy['point_u_method'], cfg_copy['line_u_method'])
            plp_associator.UpdatePL_Bipartite3d()
            plp_associator.UpdateVPLine_Bipartite3d()
            plp_associator.UpdatePlanes()
            plp_associator.UpdatePP_Bipartite3d()
            plp_associator.UpdateLP_Bipartite3d()

    img_cols = plp_associator.GetOutputImagecols()
    pointtracks = plp_associator.GetPointTracks()
    linetracks = plp_associator.GetLineTracks()
    vptracks = plp_associator.GetOutputVPTracks()
    pl_bpt3d = plp_associator.GetPL_Bipartite3d()
    vpline_bpt3d = plp_associator.GetVPLine_Bipartite3d()
    inf_planes3d = plp_associator.GetPlanes()
    pp_bpt3d = plp_associator.GetPP_Bipartite3d()
    lp_bpt3d = plp_associator.GetLP_Bipartite3d()

    return img_cols, pointtracks, linetracks, vptracks, pl_bpt3d, vpline_bpt3d, inf_planes3d, pp_bpt3d, lp_bpt3d, pl_bpt3d_before_ba, vpline_bpt3d_before_ba
