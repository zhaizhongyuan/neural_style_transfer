import numpy as np
import matplotlib.pyplot as plt

def load_data():
    content_avg = np.loadtxt('Output/content_avg_losses.csv', dtype=int, delimiter=',')
    content_max = np.loadtxt('Output/content_max_losses.csv', dtype=int, delimiter=',')
    random_avg = np.loadtxt('Output/random_avg_losses.csv', dtype=int, delimiter=',')
    random_max = np.loadtxt('Output/random_max_losses.csv', dtype=int, delimiter=',')
    style_avg = np.loadtxt('Output/style_avg_losses.csv', dtype=int, delimiter=',')
    style_max = np.loadtxt('Output/style_max_losses.csv', dtype=int, delimiter=',')
    return content_avg, content_max, random_avg, random_max, style_avg, style_max

def plot_losses(content_avg, content_max, random_avg, random_max, style_avg, style_max, method):
    plt.figure()    
    plt.plot(range(0, 5000, 25), np.log(content_avg.mean(axis=0)), 
            color = "green", linewidth = 0.5)
    plt.plot(range(0, 5000, 25), np.log(random_avg.mean(axis=0)), 
            color = "blue", linewidth = 0.5)
    plt.plot(range(0, 5000, 25), np.log(style_avg.mean(axis=0)),
            color = "red", linewidth = 0.5)
    plt.plot(range(0, 5000, 25), np.log(content_max.mean(axis=0)), 
            color = "green", linewidth = 0.5, linestyle='dashed')
    plt.plot(range(0, 5000, 25), np.log(random_max.mean(axis=0)), 
            color = "blue", linewidth = 0.5, linestyle='dashed')
    plt.plot(range(0, 5000, 25), np.log(style_max.mean(axis=0)), 
            color = "red", linewidth = 0.5, linestyle='dashed')
    if method == "all":
        labels = ["Content (Avg)", "Random (Avg)", "Style (Avg)", 
                  "Content (Max)", "Random (Max)", "Style (Max)"]
    elif method == "avg":
        labels = ["Content (Avg)", "Random (Avg)", "Style (Avg)"]
        ax = plt.gca()
        ax.set_xlim([1000, 5000])
        ax.set_ylim([10.6, 11.3])
    elif method == "max":
        labels = ["Content (Max)", "Random (Max)", "Style (Max)"]
        ax = plt.gca()
        ax.set_xlim([1000, 5000])
        ax.set_ylim([12.4, 13.1])
    plt.legend(labels = labels, loc = "upper right")
    plt.xlabel("number of epochs")
    plt.ylabel("average total loss")
    plt.title("Log Average Total Loss vs. Number of Epochs")
    plt.draw()

def main():
    content_avg, content_max, random_avg, random_max, style_avg, style_max = load_data()
    plot_losses(content_avg, content_max, random_avg, random_max, style_avg, style_max, "all")
    plot_losses(content_avg, content_max, random_avg, random_max, style_avg, style_max, "avg")
    plot_losses(content_avg, content_max, random_avg, random_max, style_avg, style_max, "max")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()