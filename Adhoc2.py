import threading
import random
import time
from confluent_kafka import Producer, Consumer


class KafkaProducerModule:
    def __init__(self, topic, bootstrap_servers):
        self.topic = topic
        self.producer = Producer({'bootstrap.servers': bootstrap_servers})

    def produce_messages(self):
        while True:
            number = random.randint(1, 100)
            self.producer.produce(self.topic, str(number))
            self.producer.flush()
            print(f"Produced: {number}")
            time.sleep(1)  # Adjust the sleep time as needed


class KafkaConsumerModule:
    def __init__(self, topic, bootstrap_servers, group_id):
        self.topic = topic
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe([self.topic])

    def consume_messages(self):
        while True:
            msg = self.consumer.poll(1.0)  # Poll for 1 second
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            print(f"Consumed: {msg.value().decode('utf-8')}")


class KafkaThreadManager:
    def __init__(self, topic, bootstrap_servers):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.producer_thread = None
        self.consumer_thread = None

    def start_threads(self):
        producer = KafkaProducerModule(self.topic, self.bootstrap_servers)
        consumer = KafkaConsumerModule(self.topic, self.bootstrap_servers, group_id="flask-group")

        self.producer_thread = threading.Thread(target=producer.produce_messages, daemon=True)
        self.consumer_thread = threading.Thread(target=consumer.consume_messages, daemon=True)

        self.producer_thread.start()
        self.consumer_thread.start()


from flask import Flask
from KafkaModule import KafkaThreadManager

app = Flask(__name__)

# Kafka configuration
TOPIC_NAME = "test-topic"
BOOTSTRAP_SERVERS = "localhost:9092"

# Initialize Kafka Thread Manager
kafka_manager = KafkaThreadManager(TOPIC_NAME, BOOTSTRAP_SERVERS)

@app.route("/")
def home():
    return "Kafka Flask Application is running!"

if __name__ == "__main__":
    # Start Kafka threads before running the Flask app
    kafka_manager.start_threads()

    # Run the Flask app
    app.run(debug=True)
    

