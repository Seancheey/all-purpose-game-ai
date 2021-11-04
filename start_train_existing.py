if __name__ == '__main__':
    from config import config

    trainer = config.provide_trainer()
    trainer.train_existing_and_save()
