start_time = time.time()    
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE * 1.5,
            epochs=2,
            layers='all')
end_time = time.time() - start_time
    print("Train model: {}".format(end_time))
