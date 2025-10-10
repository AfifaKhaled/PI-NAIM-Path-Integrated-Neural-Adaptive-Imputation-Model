# pi_naim/utils/curriculum.py
from dataclasses import dataclass


@dataclass
class CurriculumCfg:
    epochs: int
    mcar_frac: float = 0.3
    mar_frac: float = 0.4
    mnar_frac: float = 0.3


class Curriculum:
    def __init__(self, cfg: CurriculumCfg):
        self.cfg = cfg
        self.e1 = int(cfg.epochs * cfg.mcar_frac)
        self.e2 = self.e1 + int(cfg.epochs * cfg.mar_frac)

    def phase(self, epoch: int) -> str:
        if epoch < self.e1:
            return "mcar"
        if epoch < self.e2:
            return "mar"
        return "mnar"

    def get_phase_info(self, epoch: int) -> dict:
        """Get detailed information about the current phase"""
        phase = self.phase(epoch)
        info = {
            'phase': phase,
            'epoch': epoch + 1,
            'total_epochs': self.cfg.epochs,
            'progress': (epoch + 1) / self.cfg.epochs * 100
        }

        if phase == "mcar":
            info['description'] = "Missing Completely At Random"
            info['remaining'] = self.e1 - epoch
        elif phase == "mar":
            info['description'] = "Missing At Random"
            info['remaining'] = self.e2 - epoch
        else:
            info['description'] = "Missing Not At Random"
            info['remaining'] = self.cfg.epochs - epoch

        return info