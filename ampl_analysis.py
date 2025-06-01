import numpy as np
from numpy.fft import fft, ifft
from scipy.signal.windows import flattop


def czt(x: np.ndarray, m: int, A: complex, W: complex) -> np.ndarray:
    """
    Chirp Z-Transform.

    Args:
        x: 1D numpy array of shape (n,)
        m: Number of output points
        A: Complex number, start point on z-plane spiral
        W: Complex number, ratio between successive points

    Returns:
        y: 1D numpy array of shape (m,)
    """
    n = x.shape[0]
    l = int(2 ** np.ceil(np.log2(n + m - 1)))

    # Precompute chirp kernel
    k = np.arange(max(m, n), dtype=np.float64)
    w = W ** (k**2 / 2)

    # Initialize arrays
    gn = np.zeros(l, dtype=np.complex128)
    hn = np.zeros(l, dtype=np.complex128)

    # Build sequences
    a_pows = A ** (-np.arange(n, dtype=np.float64))
    gn[:n] = x * a_pows * w[:n]

    hn[:m] = 1 / w[:m]

    hn[l - n + 1:] = 1 / w[1:n][::-1]

    # Convolution via FFT
    Yk = fft(gn) * fft(hn)
    qn = ifft(Yk)

    yn = qn[:m] * w[:m]
    return yn

def sinc_interp(X, factor):
    """
    对输入 X 使用 sinc 插值，放大 factor 倍。
    
    参数:
        X (np.ndarray)
        factor (int): 插值倍数
    
    返回:
        X_interp (np.ndarray)
    """
    N = len(X)
    n = np.arange(N)
    f_new = np.linspace(0, N - 1, factor * N)

    # 利用广播机制构建差值矩阵：shape = (len(f_new), N)
    diff = f_new[:, None] - n[None, :]  # 广播实现外减法
    interp_kernel = np.sinc(diff)       # 向量化计算所有 sinc 值

    # 点乘得到插值结果
    X_interp = interp_kernel @ X

    return X_interp

class SpectrogramEngine:
    def __init__(
            self,
            audio: np.ndarray,
    ):
        self.sample_num = len(audio)
        #self.window = flattop(self.sample_num, sym=False) # 发现 flattop 效果还不如 hanning
        self.window = np.hanning(self.sample_num)
        self.audio = audio * self.window
        self.window_sum = np.sum(self.window)
        
    def get_fft_spec(
            self, 
            scale: str = 'db', 
            clamp:float = 1e-9
    ):
        spec = 2 * (np.abs(np.fft.rfft(self.audio))) / self.window_sum
        if scale == 'db':
            spec = np.log10(np.clip(spec, clamp, np.inf)) * 20

        return spec
    
    def get_zero_pad_fft_spec(
            self, 
            zero_pad_scale: int = 49, 
            scale: str = 'db', 
            clamp:float = 1e-9
    ):
        zero_pad_audio = np.concatenate([self.audio, np.zeros(zero_pad_scale * self.sample_num)])
        zero_pad_spec = 2 * (np.abs(np.fft.rfft(zero_pad_audio))) / self.window_sum
        
        if scale == 'db':
            zero_pad_spec = np.log10(np.clip(zero_pad_spec, clamp, np.inf)) * 20

        return zero_pad_spec
    
    def ampl_analysis_czt(
            self,
            f0: float,
            sr: int,
            max_nhar : int = None,
            scale: str = 'db',
            clamp: float = 1e-9  
    ): 
        nhar = int(np.floor(sr / f0 / 2))
        if max_nhar is None:
            pass
        elif nhar > max_nhar:
            nhar = max_nhar

        A = np.exp(1j * 2 * np.pi * f0 / sr)
        W = np.exp(-1j * 2 * np.pi * f0 / sr)

        yn = czt(self.audio, nhar, A, W)
        yn = 2 * yn / self.window_sum

        ampl = np.abs(yn)

        if scale == 'db':
            ampl = np.log10(np.clip(ampl, clamp, np.inf)) * 20

        return ampl
    
    