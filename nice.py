import torch
import torch.nn as nn

"""
NICEでやること
- num_couplingの数だけcoupling層を呼ぶ
- 最後のカップリング層が終わったら、スケーリング層を呼ぶ
- 対数尤度を出力する。

nn.Moduleがcallされた時の返り値はテンソルではない。
"""
class NICE(nn.Module):
    def __init__(self, prior, in_out_dim, 
                    mid_dim, num_coupling, hidden, mask_config):
        """
        prior : 事後分布
        in_out_dim : 入力と出力の次元数
        num_coupling : カップリング層の数
        mid_dim : 中間層の数(カップリング層の中のやつ)
        hidden : 隠れ層の数
        mask_config : 1->奇数番目の要素が関数mにかかる。0->偶数番目が同様に
        """
        super(NICE, self).__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim

        self.coupling = nn.ModuleList([
                            Coupling(in_out_dim=in_out_dim,
                            mid_dim=mid_dim,
                            hidden=hidden,
                            mask_config=(mask_config+i)%2) for i in range(num_coupling)
                            ])
        self.scaling = Scaling(in_out_dim)
    
    def forward(self, x, reverse=False):
        """
        reverse : True->生成、False->学習
        x : 入力データ

        返り値は、対数尤度
        """
        if reverse == False:# 学習時
            # カップリング層
            for i in range(len(self.coupling)):
                x = self.coupling[i](x, reverse)
            
            # スケーリング層
            x, log_J = self.scaling(x, reverse)
            
            # 事後分布の対数尤度
            log_prob = torch.sum(self.prior.log_prob(x), dim=1)# 各バッチごとにそれぞれの次元の対数尤度を合計する。

            # print(type(log_prob))

            lala = log_prob+log_J
            # print(lala)
            return x, log_prob + log_J
            # return x
        
        else:# サンプリング時
            # スケーリング層
            x, _ = self.scaling(x, reverse=True)

            # coupling layerを逆順で回す
            for i in reversed(range(len(self.coupling))):
                x = self.coupling[i](x, reverse=True)

            return x

"""
カップリング層でやること
- データを二つに分割
- カップリングするデータを決め
- カップリング
- データを組み合わせる

よって、入力：全データ、出力：全データ
"""
class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config, reverse=False):
        """
        in_out_dim : 入力と出力の次元数
        mid_dim : 中間層の数
        hidden : 隠れ層の数
        mask_config : どちらのデータをカップリングするか(0 or 1)
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(# 最初の隠れ層
                                nn.Linear(in_out_dim//2, mid_dim),
                                nn.ReLU())
        self.mid_block = nn.ModuleList([# 中間の隠れ層
                            nn.Sequential(
                                nn.Linear(mid_dim, mid_dim),
                                nn.ReLU()) for _ in range(hidden-1)])
        self.end_block = nn.Sequential(# 最後の層(reluは噛まない)
                                nn.Linear(mid_dim, in_out_dim//2))

    def forward(self, x, reverse):# xは、(バッチ数, 28*28)の形で入っている
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))# 前半後半で分けている(つまり、１回目は画像の上半分下半分)

        # print(x)

        if self.mask_config==0:
            a, b = x[:,:,0], x[:,:,1]# a->そのまま次の層へ, b->カップリングする
        else:
            b, a = x[:,:,0], x[:,:,1]
        
        a_ = self.in_block(a)# 最初の層
        # print(a_.size())
        # print()
        for i in range(len(self.mid_block)):# 中間層
            # print(i,"回通過")
            a_ = self.mid_block[i](a_)
        a_ = self.end_block(a_)# 最後の層

        # print(a_)

        if reverse == False:
            b = b + a_# 加法的カップリング
        else:
            b = b - a_# 生成時は、引き算になる

        if self.mask_config==0:
            x = torch.stack((a, b), dim=2)#　テンソルを積む
        else:
            x = torch.stack((b, a), dim=2)

        return x.reshape([B, W])


"""
スケーリング層でやること
- スケーリングを行う
- 対数尤度の計算
出力は、対数尤度を計算するデータ
        (スケーリング後のデータ, スケーリング層のヤコビアン)
"""
class Scaling(nn.Module):
    def __init__(self, in_out_dim):
        """
        in_out_dim : 入力と出力の数
        """
        super(Scaling, self).__init__()

        self.scale = nn.Parameter(
                        torch.zeros((1,in_out_dim)), requires_grad=True)

    def forward(self, x, reverse=False):
        """
        x : 入力データ
        reverse : 学習か生成か
        """
        log_det_J = torch.sum(self.scale)
        if reverse==False:
            x = x * torch.exp(self.scale)
        else:
            x = x * torch.exp(-self.scale)
        
        return x, log_det_J


if __name__=="__main__":
    # x = torch.randn((2,2))
    # print(x)
    # test_scaling = Scaling(2)
    # y = test_scaling(x)
    # print(x.mean())
    # print(y[0])
    # test_scaling.eval()
    # z = test_scaling(y[0])
    # print(z)

    # x = torch.randn((2,10))
    # coup = Coupling(10,20,3,1)

    # print(x)
    # y = coup(x, reverse=False)
    # # print(y)
    # z = coup(y, reverse=True)
    # print(z)


    nice_test = NICE(lambda x: x, 10, 20, 3, 3, 1)
    x = torch.randn((2,10))
    print(x)
    y = nice_test(x, reverse=False)
    z = nice_test(y, reverse=True)
    print(z)
    