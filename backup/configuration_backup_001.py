class Config:
    def __init__(self, args):
        self.data = args.data
        self.log = args.log

        self.data_name = 'yeast'#'nuswide'
        self.attri_num = 103 # 128
        self.label_list = [6, 5, 3] # [30, 20, 15] [14] [7, 6] [4, 4, 3, 2]
        self.train_instance_list = [500, 500, 500] # [1500] [800, 700] [400, 400, 400, 300] [60000, 50000, 50000]
        # self.test_instance_list = [450, 450] # [900] [250, 250, 250, 150] [300, 300, 300]
        self.task_num = len(self.label_list)

        self.first_epoch = 60
        self.joint_epoch = 4
        self.ssl_epoch = 4
        self.new_epoch = 20
        self.st_epoch = 65
        self.ts_epoch = 65

        self.first_batch = 16
        self.ssl_batch = 16
        self.new_batch = 16
        self.st_batch = 16
        self.ts_batch = 16

        self.num_workers = 16

        # self.first_epoch = 1
        # self.joint_epoch = 1
        # self.ssl_epoch = 1
        # self.new_epoch = 1
        # self.st_epoch = 1
        # self.ts_epoch = 1