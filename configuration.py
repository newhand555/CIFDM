class Config:
    def __init__(self, args):
        self.data = args.data
        self.log = args.log

        self.data_name = 'nuswide'#'yeast'
        self.attri_num = 128 # 128
        self.embed_dim = 64
        self.label_list = [30, 26, 25] #[6, 5, 3]#  [14] [7, 6] [4, 4, 3, 2]
        self.train_instance_list = [60000, 50000, 50000] # [500, 500, 500] # [1500] [800, 700] [400, 400, 400, 300]
        # self.test_instance_list = [450, 450] # [900] [250, 250, 250, 150] [300, 300, 300]
        self.task_num = len(self.label_list)

        self.first_epoch = 1
        self.joint_epoch = 1
        self.ssl_epoch = 1
        self.new_epoch = 1
        self.st_epoch = 1
        self.ts_epoch = 1

        self.first_batch = 64
        self.ssl_batch = 64
        self.new_batch = 64
        self.st_batch = 64
        self.ts_batch = 64
        self.eval_batch = 256

        self.num_workers = 24
        self.weight = 5