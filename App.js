import React, { useState, useEffect } from "react";
import { StyleSheet, Text, View, Image, TouchableOpacity } from "react-native";
import * as FileSystem from "expo-file-system";
import * as ImagePicker from "expo-image-picker";
import * as Speech from "expo-speech";
import * as tf from "@tensorflow/tfjs";
import { decodeJpeg } from "@tensorflow/tfjs-react-native";
import * as mobilenet from "@tensorflow-models/mobilenet";

export default function App() {
  const [isTfReady, setIsTfReady] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [image, setImage] = useState(null);
  const [model, setModel] = useState(null);

  useEffect(() => {
    async function loadTfAndModel() {
      await tf.ready();
      setIsTfReady(true);
      const model = await mobilenet.load();
      setModel(model);
    }
    loadTfAndModel();
  }, []);

  useEffect(() => {
    if (predictions) {
      Speech.speak("I think it may be");
      predictions.map((prediction) => Speech.speak(prediction.className));
    }
  }, [predictions]);

  classifyImage = async () => {
    try {
      const imageAssetPath = Image.resolveAssetSource(image);
      const imgB64 = await FileSystem.readAsStringAsync(imageAssetPath.uri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = tf.util.encodeString(imgB64, "base64").buffer;
      const raw = new Uint8Array(imgBuffer);
      const imageTensor = decodeJpeg(raw);
      const predictions = await model.classify(imageTensor);
      setPredictions(predictions);
      console.log(predictions);
    } catch (error) {
      console.log(error);
    }
  };

  selectImageFromLibrary = async () => {
    try {
      setPredictions(null);
      setImage(null);
      let response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
      });

      if (!response.cancelled) {
        const source = { uri: response.uri };
        setImage(source);
        classifyImage();
      }
    } catch (error) {
      console.log(error);
    }
  };

  takePhoto = async () => {
    try {
      setPredictions(null);
      setImage(null);
      let response = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
      });

      if (!response.cancelled) {
        const source = { uri: response.uri };
        setImage(source);
        classifyImage();
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.loadingContainer}>
        <Text style={styles.text}>
          {isTfReady && model ? "Loaded" : "Loading..."}
        </Text>
      </View>
      <View style={styles.imageContainer}>
        {image && <Image source={image} style={styles.image} />}
      </View>
      {model && (
        <View>
          <TouchableOpacity
            style={styles.button}
            onPress={selectImageFromLibrary}
          >
            <Text>Choose image</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={takePhoto}>
            <Text>Take Photo</Text>
          </TouchableOpacity>
        </View>
      )}
      <View style={styles.predictionContainer}>
        {model && image && (
          <Text style={styles.text}>
            Predictions: {predictions ? "" : "Predicting..."}
          </Text>
        )}
        {model &&
          predictions &&
          predictions.map((prediction) => (
            <Text key={prediction.className} style={styles.text}>
              {prediction.className}
            </Text>
          ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#171f24",
    alignItems: "center",
  },
  loadingContainer: {
    marginTop: 50,
  },
  text: {
    color: "#ffffff",
    fontSize: 16,
    textAlign: "center",
  },
  imageContainer: {
    width: 280,
    height: 280,
    padding: 10,
    margin: 20,
    borderColor: "#cf667f",
    borderWidth: 5,
    borderStyle: "dashed",
    justifyContent: "center",
    alignItems: "center",
  },
  image: {
    width: 250,
    height: 250,
  },
  predictionContainer: {
    height: 100,
    width: "100%",
    flexDirection: "column",
    alignItems: "center",
  },
  button: {
    padding: 20,
    margin: 10,
    borderRadius: 5,
    backgroundColor: "#cf657f",
    alignItems: "center",
  },
});
