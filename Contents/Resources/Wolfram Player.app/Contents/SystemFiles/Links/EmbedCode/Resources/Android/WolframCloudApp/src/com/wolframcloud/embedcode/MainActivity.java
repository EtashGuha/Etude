package com.wolframcloud.embedcode;

import java.util.ArrayList;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.ImageView;

public class MainActivity extends Activity implements CloudInterface {

	// UI Elements
	`fields`
	private TextView resultTextView;
	private ImageView resultImageView;
	private ProgressBar bar;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		// Standard onCreate calls
		super.onCreate(savedInstanceState);
		setContentView(R.layout.main);

		// Link UI elements to Java objects
		`fieldsInitialization`
		resultTextView = (TextView) findViewById(R.id.textView);
		resultImageView = (ImageView) findViewById(R.id.imageView);
		bar = (ProgressBar) findViewById(R.id.progressBar);
	}

	public void buttonClick(View v) {
		// Make progress bar visible to show user the app is working
		bar.setVisibility(View.VISIBLE);

		// Clear previous output
		resultTextView.setText("");
		resultImageView.setImageResource(android.R.color.transparent);

		// Dismiss the keyboard
		InputMethodManager imm =(InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
		imm.hideSoftInputFromWindow(field0.getWindowToken(), 0);

		// Get the text entered by the user into fields
		`getInputsFromFields`

		// Create a CloudCompute object, defined below.
		// CloudCompute is a class that contains all of the code that interacts with Wolfram Cloud
		CloudCompute cc = new CloudCompute(this);

		// This runs the code contained in the doInBackground method of CloudCompute
        ArrayList<String> list = new ArrayList<String>();
        `addInputsToList`
        cc.execute(list);
	}

	@Override
	public void onConnectCompleted(boolean b) {}

	@Override
	public void onEvaluateCompleted(String result) {
		// Make progress bar invisible again
		bar.setVisibility(View.GONE);

		// Display result
		resultTextView.setText(result);
	}

	@Override
	public void onFailed(){}
	
	@Override
	public String getBaseURL(){
		return "`url`";
	}
}
