import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from skimage import io
import glob
import time
from track.track_utils import parse_args
from track.sort_tracker import Sort
from src.utils.logger import log

matplotlib.use("TkAgg")

np.random.seed(0)

if __name__ == "__main__":
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if display:
        if not os.path.exists("mot_benchmark"):
            log.error("\n\t'mot_benchmark' link not found!\n\n    \
                    Create a symbolic link to the MOT benchmark\n    \
                        (https://motchallenge.net/data/2D_MOT_2015/#download).\
                        E.g.:\n\n  \
                        $ ln -s /path/to/MOT2015_challenge/2DMOT2015 \
                            mot_benchmark\n\n")
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect="equal")

    if not os.path.exists("output"):
        os.makedirs("output")
    pattern = os.path.join(args.seq_path, phase, "*", "det", "det.txt")
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold
                           )  # create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=",")
        seq = seq_dets_fn[pattern.find("*"):].split(os.path.sep)[0]

        with open(os.path.join("output", "%s.txt" % (seq)), "w") as out_file:
            log.info("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                dets[:,
                     2:4] += dets[:, 0:
                                  2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if display:
                    fn = os.path.join("mot_benchmark", phase, seq, "img1",
                                      "%06d.jpg" % (frame))
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + " Tracked Targets")

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    log.info(
                        "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" %
                        (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                        file=out_file,
                    )
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(
                            patches.Rectangle((d[0], d[1]),
                                              d[2] - d[0],
                                              d[3] - d[1],
                                              fill=False,
                                              lw=3,
                                              ec=colours[d[4] % 32, :]))

                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    log.info("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" %
             (total_time, total_frames, total_frames / total_time))

    if display:
        log.info("Note: to get real runtime results run without the option: \
                --display")
