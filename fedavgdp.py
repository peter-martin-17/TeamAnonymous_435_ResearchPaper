import tensorflow as tf
import tensorflow_federated as tff
# import tensorflow_privacy as tfp

source, _ = tff.simulation.datasets.emnist.load_data()

def client_data(n: int) -> tf.data.Dataset:
    return source.create_tf_dataset_for_client(source.client_ids[n]).map(
        lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
    ).repeat(10).batch(20)

train_data = [client_data(n) for n in range(3)]

def model_fn() -> tff.learning.Model:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, tf.nn.softmax, input_shape=(784,),
        kernel_initializer='zeros')
    ])
    return tff.learning.from_keras_model(
        model,
        input_spec=train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# query = tfp.QuantileAdaptiveClipSumQuery(
#     initial_l2_norm_clip=0.01,
#     noise_multiplier=5,
#     target_unclipped_quantile=0.5,
#     learning_rate=0.2,
#     clipped_count_stddev=40712.75,
#     expected_num_records=814255
#     )

# query = tfp.NormalizedQuery(query, denominator=5)

# model_update_aggregator = tff.aggregators.DifferentiallyPrivateFactory(
#     query
# )

# trainer = tff.learning.build_federated_averaging_process(
#     model_fn,
#     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
#     model_update_aggregation_factory = model_update_aggregator)

trainer = tff.learning.build_federated_averaging_process(
     model_fn,
     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))

state = trainer.initialize()
for _ in range(5):
    state, metrics = trainer.next(state, train_data)
    print(metrics['train']['loss'])
