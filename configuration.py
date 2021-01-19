class Config:
    def __init__(self, args):
        self.data = args.data
        self.log = args.log

        self.data_name = 'nuswide'#'yelp''yeast'
        self.attri_num = 128#668 # 128
        self.embed_dim = 64
        self.label_list = [20, 20, 20, 20]##  [14] [7, 6] [4, 4, 3, 2][2, 2, 2][6, 5, 3]
        self.train_instance_list = [40000, 40000, 40000, 40000]##[2800, 2800, 2800] #   [1500] [800, 700] [400, 400, 400, 300][500, 500, 500]
        # self.test_instance_list = [450, 450] # [900] [250, 250, 250, 150] [300, 300, 300]
        self.task_num = len(self.label_list)

        self.first_epoch = 5
        self.pse_epoch = 1
        self.ssl_epoch = 5
        self.new_epoch = 30
        self.sti_epoch = 5
        self.ste_epoch = 30
        self.ts_epoch = 30
        self.as_epoch = 30

        self.first_batch = 128
        self.ssl_batch = 128
        self.new_batch = 128
        self.st_batch = 128
        self.ts_batch = 128
        self.as_batch = 128
        self.eval_batch = 256

        self.num_workers = 24
        self.weight = 5