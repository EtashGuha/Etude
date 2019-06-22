package com.wolframcloud.embedcode;

public interface CloudInterface {
	public void onConnectCompleted(boolean b);
	public void onEvaluateCompleted(String result);
	public void onFailed();
	public String getBaseURL();
}
