package com.wolframcloud.embedcode;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;

import android.content.Context;
import android.os.AsyncTask;
import android.content.Context;
import android.os.AsyncTask;

// CloudCompute is an asynchronous task that takes the first item in <> and returns the third item
public class CloudCompute extends AsyncTask<List<String>, Void, String> {

	// URL of Cloud API we want to use
	private String baseURL = "";

	// Callback instance of a CloudInterface. This allows us to call
	// methods in MainActivity (which implements CloudInterface).
	private CloudInterface callback;

	// Constructor
	public CloudCompute(Context c) {
		// This links the MainActivity context to the callback variable created above
		callback = (CloudInterface) c;
		baseURL = callback.getBaseURL();
	}

	// This is the method that does most of the work. It is run by 
	// calling the execute command in MainActivity. It takes any number
	//  of String Lists as its parameter.
	@Override
	protected String doInBackground(List<String>... params) {
		// Initialize the output object
		String resultString = "";

		// For each parameter object given...
		for (List<String> list : params) {
			try {
				// Add parameters to the URL
				baseURL += "?";
				`outputWrite`
				
				// Create a URL object with the base URL address
				URL url = new URL(baseURL);
				
				// Create an input reader to get the Cloud API's output
				BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()));
				
				// Save the Cloud API's output
				String outLine; while ((outLine = in.readLine()) != null) {resultString += outLine;}

				// Close the stream and connection
				in.close();
				
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return resultString;
	}

	// This method runs after doInBackground is finished, automatically using
	// doInBackground's output as a parameter
	@Override
	protected void onPostExecute(String result) {
		callback.onEvaluateCompleted(result);
	}
}
