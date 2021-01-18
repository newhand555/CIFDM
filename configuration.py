class Config:
    def __init__(self, args):
        self.data = args.data
        self.log = args.log

        self.data_name = 'yelp'#'yeast'#'nuswide'
        self.attri_num = 668 # 128
        self.embed_dim = 64
        self.label_list = [2, 2, 2]#[30, 26, 25] #  [14] [7, 6] [4, 4, 3, 2][6, 5, 3]
        self.train_instance_list = [2800, 2800, 2800] #  [60000, 50000, 50000]# [1500] [800, 700] [400, 400, 400, 300][500, 500, 500]
        # self.test_instance_list = [450, 450] # [900] [250, 250, 250, 150] [300, 300, 300]
        self.task_num = len(self.label_list)

        self.first_epoch = 20
        self.joint_epoch = 20
        self.ssl_epoch = 5
        self.new_epoch = 5
        self.st_epoch = 60
        self.ts_epoch = 60

        self.first_batch = 64
        self.ssl_batch = 64
        self.new_batch = 64
        self.st_batch = 64
        self.ts_batch = 64
        self.eval_batch = 256

        self.num_workers = 24
        self.weight = 5