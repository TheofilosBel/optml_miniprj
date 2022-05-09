from typing import Any
from torch.utils.tensorboard import SummaryWriter
from src.utils import create_tb_dir
import os

class TBWritter:
    def __init__(self,args) -> None:
        self.tb_writer = None
        self.args = args

    def _lazy_load(self,):
        tb_dir = create_tb_dir(self.args)
        self.tb_writer = SummaryWriter(tb_dir)
        self.file = open(os.path.join(tb_dir, "run.txt"), "w+")
        self.fp_closed_err = False

    def writer(self) -> SummaryWriter:
        if self.tb_writer == None:
            self._lazy_load()
        return self.tb_writer

    def close(self):
        if self.tb_writer: self.tb_writer.close()
        self.file.close()

    def write_to_file(self, msg:str):
        if self.file.closed:
            if self.fp_closed_err == False:
                print('[ERR] result file ptr closed...')
                self.fp_closed_err = True
            return

        self.file.write(msg + "\n")
        self.file.flush()


def tb_write_metrics(
    args: Any,
    tbwriter: TBWritter,
    errD: float,
    errG: float,
    epoch: int,
    cur_iter_ctr: int,
    tot_iter_ctr: int,
    tot_dl_size: int,
    D_x:float, D_G_z1:float, D_G_z2:float
) -> None :
    '''
        Writes metrics to tbwriter and return them
    '''
    # TB is lazily intied here to avoid creating multiple dir when bugs exist and program doesn't run

    # Write losses to tb
    tbwriter.writer().add_scalar(f"Loss/Loss_D", errD, tot_iter_ctr)
    tbwriter.writer().add_scalar(f"Loss/Loss_G", errG, tot_iter_ctr)

    # Write losses to file:
    msg = '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' \
        % (epoch, args.num_epochs, cur_iter_ctr, tot_dl_size, errD, errG, D_x, D_G_z1, D_G_z2)
    tbwriter.write_to_file(msg)

    return
