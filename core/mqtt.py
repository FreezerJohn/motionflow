"""
MQTT Publisher for MotionFlow events.

This module provides an MQTT client that publishes events to a broker.
It implements the EventListener interface so it can be registered with EventManager.

The publisher is optional - if paho-mqtt is not installed or MQTT is disabled,
the system continues to work without it.
"""

import json
import logging
from typing import TYPE_CHECKING

from core.events import EventListener, MotionEvent

# Optional import - MQTT is not required
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None

if TYPE_CHECKING:
    from config.schema import MqttSettings


class MqttPublisher(EventListener):
    """
    Publishes MotionFlow events to an MQTT broker.

    Topic structure: {base_topic}/{stream_id}/{zone_id}

    Example topics:
        motionflow/living_room/couch
        motionflow/kitchen/door

    Events are published as JSON with the structure:
        {
            "event": "enter",
            "timestamp": 1702512345.678,
            "person": {"id": 1, "action": "sitting", "direction": "left"},
            "occupancy": 1
        }
    """

    def __init__(self, settings: 'MqttSettings'):
        """
        Initialize the MQTT publisher.

        Args:
            settings: MqttSettings from the config
        """
        if not MQTT_AVAILABLE:
            raise ImportError(
                "paho-mqtt is not installed. Install it with: pip install paho-mqtt"
            )

        self.settings = settings
        self.base_topic = settings.base_topic
        self.qos = settings.qos
        self.logger = logging.getLogger('mqtt')

        # Create MQTT client
        self.client = mqtt.Client(
            client_id=f"motionflow-{id(self)}",
            protocol=mqtt.MQTTv5
        )

        # Set up authentication if provided
        if settings.username and settings.password:
            self.client.username_pw_set(settings.username, settings.password)

        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self._connected = False
        self._broker = settings.broker
        self._port = settings.port

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """Callback when connected to broker."""
        if reason_code == 0:
            self._connected = True
            self.logger.info(f"Connected to MQTT broker at {self._broker}:{self._port}")
        else:
            self.logger.error(f"Failed to connect to MQTT broker: {reason_code}")

    def _on_disconnect(self, client, userdata, reason_code, properties=None):
        """Callback when disconnected from broker."""
        self._connected = False
        if reason_code != 0:
            self.logger.warning(f"Disconnected from MQTT broker: {reason_code}")

    def start(self) -> None:
        """Connect to the MQTT broker."""
        try:
            self.client.connect(self._broker, self._port, keepalive=60)
            self.client.loop_start()
            self.logger.info(f"Connecting to MQTT broker at {self._broker}:{self._port}...")
        except Exception as e:
            self.logger.error(f"Failed to connect to MQTT broker: {e}")

    def stop(self) -> None:
        """Disconnect from the MQTT broker."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            self.logger.info("Disconnected from MQTT broker")
        except Exception as e:
            self.logger.error(f"Error disconnecting from MQTT broker: {e}")

    def on_event(self, event: MotionEvent) -> None:
        """
        Publish an event to MQTT.

        Zone-related events are published to: {base_topic}/{stream_id}/{zone_id}
        Track-level events are published to: {base_topic}/{stream_id}
        """
        if not self._connected:
            return

        # Determine topic
        if event.zone_id:
            topic = f"{self.base_topic}/{event.stream_id}/{event.zone_id}"
        else:
            topic = f"{self.base_topic}/{event.stream_id}"

        # Build payload
        payload = event.to_dict()

        # Publish
        try:
            result = self.client.publish(
                topic,
                json.dumps(payload),
                qos=self.qos
            )
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                self.logger.warning(f"Failed to publish to {topic}: {result.rc}")
            else:
                self.logger.debug(f"Published to {topic}: {payload}")
        except Exception as e:
            self.logger.error(f"Error publishing to MQTT: {e}")


def create_mqtt_publisher(settings: 'MqttSettings') -> MqttPublisher | None:
    """
    Factory function to create an MQTT publisher if available and enabled.

    Returns None if MQTT is disabled or paho-mqtt is not installed.
    This allows the system to run without MQTT support.

    Args:
        settings: MqttSettings from the config

    Returns:
        MqttPublisher instance or None
    """
    if not settings.enabled:
        logging.info("MQTT publishing is disabled")
        return None

    if not MQTT_AVAILABLE:
        logging.warning(
            "MQTT is enabled but paho-mqtt is not installed. "
            "Install it with: pip install paho-mqtt"
        )
        return None

    try:
        publisher = MqttPublisher(settings)
        return publisher
    except Exception as e:
        logging.error(f"Failed to create MQTT publisher: {e}")
        return None
