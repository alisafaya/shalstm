from pytorch_lamb import Lamb

class MinTrustLamb(Lamb):

    def __init__(self, *args, **kwargs):
        super(MinTrustLamb, self).__init__(min_trust=0.25, *args, **kwargs)