class Config:
    def __init__(self, args):
        self.data = args.data
        self.log = args.log

        self.data_name ='mirflickr' #'nuswide''yelp''yeast'
        self.attri_num = 1000#668 # 128
        self.embed_dim = 128
        self.label_list = [11, 9, 9, 9]#[20, 20, 20, 20]#  [14] [7, 6] [4, 4, 3, 2][2, 2, 2][7, 4, 3]
        self.train_instance_list = [2500, 2500, 2500, 2500]#[40000, 40000, 40000, 40000]#[2800, 2800, 2800] #   [1500] [800, 700] [400, 400, 400, 300][500, 500, 500]
        # self.test_instance_list = [450, 450] # [900] [250, 250, 250, 150] [300, 300, 300]
        self.task_num = len(self.label_list)

        self.first_epoch = 20
        self.pse_epoch = 1
        self.ssl_epoch = 1
        self.new_epoch = 20
        self.sti_epoch = 5
        self.ste_epoch = 20
        self.ts_epoch = 20
        self.as_epoch = 20

        self.first_batch = 64
        self.ssl_batch = 64
        self.new_batch = 64
        self.st_batch = 64
        self.ts_batch = 64
        self.as_batch = 64
        self.eval_batch = 256

        self.num_workers = 24
        self.weight = 30
        self.use_teacher = False
        self.gamma = 8

    def __str__(self):
        result = 'data_name: {}.\nembed_dim: {}.\ntask_num: {}.\nweight: {}.\nuse_teacher: {}.\ngamma: {}.\n'.format(self.data_name, self.embed_dim, self.task_num, self.weight, self.use_teacher, self.gamma)
        result += 'first_epoch: {}.\npse_epoch: {}.\nssl_epoch {}.\nnew_epoch: {}.\nsti_epoch: {}.\nste_epoch: {}.\nts_epoch: {}.\nas_epoch: {}\n'
        result += 'first_batch: {}.\nssl_batch: {}.\nnew_batch: {}.\nst_batch: {}.\nts_batch: {}.\nas_batch: {}\n'
        return  result