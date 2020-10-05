class Config:
    def __init__(self, args):
        self.data = args.data
        self.log = args.log

        self.data_name = 'yeast'
        self.attri_num = 103
        self.label_list = [4, 3, 3] # [14] [6, 5, 3]
        self.train_instance_list = [500, 500, 500] # [1500] [500, 500, 500]
        self.test_instance_list = [300, 300, 300] # [900] [300, 300, 300]
        self.task_num = len(self.label_list)

        self.first_epoch = 40
        self.joint_epoch = 4
        self.ssl_epoch = 0
        self.new_epoch = 16
        self.st_epoch = 65
        self.ts_epoch = 65

        self.first_batch = 4
        self.ssl_batch = 4
        self.new_batch = 4
        self.st_batch = 4
        self.ts_batch = 4

        # self.first_epoch = 1
        # self.joint_epoch = 1
        # self.ssl_epoch = 1
        # self.new_epoch = 1
        # self.st_epoch = 1
        # self.ts_epoch = 1