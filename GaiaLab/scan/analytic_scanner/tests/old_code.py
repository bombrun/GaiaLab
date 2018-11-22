        self.N_ss = np.zeros((num_sources*5, num_sources*5))  # 5 source params
        self.N_aa = np.zeros((4, 4))  # 4 attitude params  # WARNING: not the correct shpe

        self.init_blocks()

    def init_blocks(self):
        """
        Initialize the block
        """
        if self.verbose:
            print('initializing N_ss of shape: {}'.format(self.N_ss.shape))
            print('initializing N_aa of shape: {}'.format(self.N_aa.shape))
        self.__init_N_ss()
        self.__init_N_aa()

    def __init_N_ss(self):
        """ initialize the matrix N_ss """

        for i in range(0, self.N_ss.shape[0], 5):  # Nss is symmetric and square
            dR_ds = self.dR_ds(i)  # i being the source index
            # W = np.eye(5)  # TODO: implement the weighting factor
            self.N_ss[i*5:i*5+5, i*5:i*5+5] = dR_ds.transpose() @ dR_ds  # @ W  # should we use np.sum?
            # The rest of N_ss are zero by initialisation

    def __init_N_aa(self):
        """
        Initialize the matrix N_aa
        N_aa
        for n in range(0, self.N_aa.shape[0], 4):
        """
        pass
