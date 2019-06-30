/*
 * @(#)InvokeEngineEvaluateRunnable.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.concurrent;

import com.wolfram.bsf.util.MathematicaBSFException;
import com.wolfram.bsf.util.concurrent.InvokeResultRunnable;

import com.wolfram.guikit.GUIKitEnvironment;
import com.wolfram.guikit.GUIKitException;

/**
 * InvokeEngineEvaluateRunnable
 */
public class InvokeEngineEvaluateRunnable extends InvokeResultRunnable {
  
    private GUIKitEnvironment env;
    private String evalContext;
    private String scriptString;

    public InvokeEngineEvaluateRunnable(GUIKitEnvironment env, String evalContext, String scriptString) {
      this.env = env;
      this.evalContext = evalContext;
      this.scriptString = scriptString;
      }
      
    public void run() {
      Object result = null;
      try {
        result = env.evaluateScript(evalContext, scriptString);
        }
      catch (GUIKitException be) {
        exception = be;
        }
      finally {
        setResult(result);
        }
      cleanup();
      }
      
    public void cleanup(){
      env = null;
      evalContext = null;
      scriptString = null;
      }
    
	public void handleException() throws Exception {
		if (exception != null) {
			if (exception instanceof MathematicaBSFException)
				((MathematicaBSFException)exception).printStackTrace();
			}
		}
		
    }