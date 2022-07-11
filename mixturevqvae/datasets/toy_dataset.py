import numpy as np
import torch
import matplotlib.pyplot as plt

class ToyDataset():
    def __init__(self, N=1000, K=8, d=5, batch_size=32):
        self.K=K
        self.d=d
        self.centroids = self.get_centroids()
        self.np_data = self.generate_toy_data(N, K, d)
        self.data_torch = torch.tensor(self.np_data).type(torch.float32)
        self.dataloader = torch.utils.data.DataLoader(self.data_torch, batch_size)

    def get_centroids(self):
        centroids = [[np.cos(x* np.pi * 2.0/self.K), np.sin(x* np.pi*2.0/self.K)] for x in range(self.K)]
        return np.array(centroids)

    def generate_toy_data(self, N, K, d):
        # random uniform
        X = np.random.randint(0,K, size=(N,d))
        for i in range(1,d):
            plus_minus_one = np.random.randint(0,2, size=(N))*2-1
            X[:,i] = (X[:,i-1] + plus_minus_one) % K
        noisex = np.random.normal(size=(N,d)) * 0.05
        noisey = np.random.normal(size=(N,d)) * 0.05
        x_centers,y_centers = [np.cos(X* np.pi/4.0) + noisex, np.sin(X* np.pi/4.0) + noisey]
        return np.stack((x_centers,y_centers)).transpose(1,2,0)

    def discretize(self, x):
        dot = x @ self.centroids.T
        dists = 1 + np.expand_dims(np.linalg.norm(x, axis=-1),-1) - 2* dot
        return np.argmin(dists, axis=-1)

    def eval_samples(self, db, n_samples, index = -1):
        total = 0.0

        def eval_seq(x):
            diffs = (x[1:]-x[:-1])%10
            return ((diffs==1) + (diffs==8)).mean()

        for n in range(n_samples):
            trajs = db.sample((1,5),trajectory=True)
            res = self.discretize(trajs[index])
            total += eval_seq(res)
        return total / n_samples

    def plot_eval_depending_on_t(self, db, steps, n_samples=10):
        efficiency = [self.eval_samples(db, n_samples, index) for index in range(1,steps,5)]
        plt.plot(list(range(1,steps,5)), efficiency)
        plt.show()

    def display_data(self):
        plt.figure(figsize=(8,8))
        # display data
        plt.scatter(self.np_data[:,0,0], self.np_data[:,0,1], s=5, c='black', alpha=0.3);
        # display circle
        circle = plt.Circle((0, 0), 1.0, color='b', fill=False, ls='--')
        plt.gca().add_patch(circle)
        plt.axis("off")

        # Annotate centroids
        bbox_props = dict(boxstyle="square,pad=0.3", ec='black', fc='white',
                              lw=1, alpha=0.9)
        for t in range(8):
            plt.annotate(t+1, xy=(self.centroids[t,0]*1.2, self.centroids[t,1]*1.2),
                        xycoords='data',
                        xytext=(0.05 + np.random.uniform(-0.2, 0.2),
                                0.05 + np.random.uniform(-0.2, 0.2)),
                        textcoords='offset points',
                        bbox=bbox_props)

        # display 2 trajectories
        # for i in range(4):
        #     plt.plot(self.np_data[6,i:i+2,0], self.np_data[6,i:i+2,1], color=plt.cm.plasma(i/5), alpha=0.8)
        for i in range(4):
            plt.plot(self.np_data[2,i:i+2,0], self.np_data[2,i:i+2,1], color=plt.cm.plasma(i/5), alpha=0.8)
        plt.scatter(self.np_data[2,:,0], self.np_data[2,:,1], s=5, c='r');
        plt.scatter(self.np_data[6,:,0], self.np_data[6,:,1], s=5, c='r');

        plt.xlim((-1.2, 1.2))
        plt.ylim((-1.2, 1.2))
        # plt.show()

    def display_embedding(self, embedding):
        plt.figure(figsize=(5,5))
        circle = plt.Circle((0, 0), 1.0, color='b', fill=False, ls='--')
        plt.gca().add_patch(circle)
        plt.axis("off")
        bbox_props = dict(boxstyle="square,pad=0.3", ec='black', fc='white',
                              lw=1, alpha=0.9)
        plt.scatter(embedding[:,0], embedding[:,1], c="black", s=10)
        for t in range(embedding.shape[0]):
            plt.annotate(t+1, xy=(embedding[t,0]*1.2, embedding[t,1]*1.2),
                        xycoords='data',
                        xytext=(0.05 + np.random.uniform(-0.2, 0.2),
                                0.05 + np.random.uniform(-0.2, 0.2)),
                        textcoords='offset points',
                        bbox=bbox_props)
        plt.xlim((-1.2, 1.2))
        plt.ylim((-1.2, 1.2))
        plt.show();

    def display_traj(self, pts, annotate="final"):
        n_pts, n_traj, dim = pts.shape
        bbox_props = dict(boxstyle="square,pad=0.3", ec='black', fc='white',
                          lw=1, alpha=0.5)
        plt.figure(figsize=(5,5))
        for t in range(n_traj):
            for i in range(n_pts-1):
                plt.plot(pts[i:i+2,t,0], pts[i:i+2,t,1], color=plt.cm.plasma(i/n_pts), alpha=0.8)

                if annotate=="middle":
                    if i%10==9:
                        plt.annotate(i, xy=(pts[i,t,0], pts[i,t,1]),
                                    xycoords='data',
                                    xytext=(0.05 + np.random.uniform(-0.2, 0.2),
                                            0.05 + np.random.uniform(-0.2, 0.2)),
                                    textcoords='offset points',
                                    bbox=bbox_props)
            if annotate=="final":
                plt.annotate(t+1, xy=(pts[-1,t,0], pts[-1,t,1]),
                            xycoords='data',
                            xytext=(0.05 + np.random.uniform(-0.2, 0.2),
                                    0.05 + np.random.uniform(-0.2, 0.2)),
                            textcoords='offset points',
                            bbox=bbox_props)
            elif annotate=="start":
                plt.annotate(t+1, xy=(pts[0,t,0], pts[0,t,1]),
                            xycoords='data',
                            xytext=(0.05 + np.random.uniform(-0.2, 0.2),
                                    0.05 + np.random.uniform(-0.2, 0.2)),
                            textcoords='offset points',
                            bbox=bbox_props)

        circle = plt.Circle((0, 0), 1.0, color='b', fill=False, ls='--')
        plt.gca().add_patch(circle)
        plt.axis('off')
        plt.xlim((-1.1, 1.1))
        plt.ylim((-1.1, 1.1));
