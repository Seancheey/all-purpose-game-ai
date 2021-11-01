if __name__ == '__main__':
    from config import config

    visualizer = config.provide_data_visualizer()
    visualizer.visualize_all()
