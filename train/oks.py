import numpy as np


class ObjectKeyPointSimilarity:
    def __init__(self):
        self.oks_score = 0.0
        self.oks_items = 0.0
        self.k_constant = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72,
                                    .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    def update_state(self, y_true, y_pred, segmentation_area):
        g = y_true.numpy()
        xg = g[:, 0::3]
        yg = g[:, 1::3]
        vg = g[:, 2::3]

        s = segmentation_area.numpy()

        d = y_pred.numpy()
        xd = d[:, 0::3]
        yd = d[:, 1::3]
        vd = d[:, 2::3]

        dx = (xd - xg)
        dy = (yd - yg)

        dx = dx * vg
        dy = dy * vg

        k = self.k_constant
        e = (dx ** 2 + dy ** 2) / (2 * (s + np.spacing(1)) * (k ** 2))
        e = e[vg > 0]
        score = np.exp(-e)
        self.oks_score += np.sum(score)
        self.oks_items += score.shape[0]
        pass

    def reset_states(self):
        self.oks_score = 0
        self.oks_items = 0

    def result(self):
        if self.oks_items > 0:
            return self.oks_score / self.oks_items
        else:
            return 0
