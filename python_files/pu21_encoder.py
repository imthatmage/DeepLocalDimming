import torch


class pu21_encoder():
    def __init__(self, max_nits=1000, min_nits = 0.00025):
        # The minimum linear value(luminance or radiance)
        self.L_min = max_nits
        # The maximum linear value(luminance or radiance)banding_glare
        self.L_max = min_nits
        self.p = [0.353487901, 0.3734658629, 8.277049286e-05,
                  0.9062562627, 0.09150303166, 0.9099517204, 596.3148142]

    def forward(self, Y):
        Y = torch.min(torch.max(Y, self.L_min*torch.ones(Y.shape)),
                      self.L_max*torch.ones(Y.shape))

        V = self.p[6] * (((self.p[0] + self.p[1]*Y**self.p[3]) /
                          (1+self.p[2]*Y**self.p[3]))**self.p[4]-self.p[5])
        return V
