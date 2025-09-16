import config.config_vae as c


n_imgs = 4
n_plots = 2
figsize = (4,4)

class Visualizer:
    def __init__(self, loss_labels):
            self.n_losses = len(loss_labels)
            self.loss_labels = loss_labels

            header = 'Epoch'
            for l in loss_labels:
                header += '\t\t%s' % (l)

            self.config_str = ""
            self.config_str += "==="*30 + "\n"
            self.config_str += "Config options:\n\n"

            for v in dir(c):
                if v[0]=='_': continue
                s=eval('c.%s'%(v))
                self.config_str += "  {:25}\t{}\n".format(v,s)

            self.config_str += "==="*30 + "\n"

            # print(self.config_str)
            # print(header)

    def update_losses(self, losses, epoch, iter, *args):
        print('\r', '    '*20, end='')
        line = '\rEpoch: %.3i(Iter: %.3i)' % (epoch, iter)
        for l in losses[:-1]:
            line += '\t|\t%.4f' % (l)
        line += f'\t|\t{losses[-1]}'
        
        print(line)

    def update_images(self, *img_list):
        pass

    def update_hist(self, *args):
        pass

    def update_running(self, *args):
        pass


visualizer = Visualizer(c.loss_names)

def show_loss(losses, epoch, iter, logscale=False):
    visualizer.update_losses(losses, epoch, iter)

def show_imgs(*imgs):
    visualizer.update_images(*imgs)

def show_hist(data):
    visualizer.update_hist(data.data)

def signal_start():
    visualizer.update_running(True)

def signal_stop():
    visualizer.update_running(False)

def close():
    visualizer.close()

