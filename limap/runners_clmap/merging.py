import _limap._clmap as _clmap


def remerge_clmap(linker3d, linetracks, num_outliers=2, use_collinearity2D=True, overlap_th=0.0, angle_th=2.0, perp_dist_th=2.0):
    if len(linetracks) == 0:
        return linetracks
    new_linetracks = linetracks
    num_tracks = len(new_linetracks)
    # iterative remerging
    while True:
        new_linetracks = _clmap.RemergeLineTracks(
            new_linetracks, linker3d, num_outliers=num_outliers, use_collinearity2D=use_collinearity2D, overlap_th=overlap_th, angle_th=angle_th, perp_dist_th=perp_dist_th)
        num_tracks_new = len(new_linetracks)
        if num_tracks == num_tracks_new:
            break
        num_tracks = num_tracks_new
    print("[LOG] tracks after iterative remerging: {0} / {1}".format(len(new_linetracks), len(linetracks)))
    return new_linetracks
