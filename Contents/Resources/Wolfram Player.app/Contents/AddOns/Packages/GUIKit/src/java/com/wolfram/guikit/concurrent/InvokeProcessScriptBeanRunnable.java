/*
 * @(#)InvokeProcessScriptBeanRunnable.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.concurrent;

import java.net.URL;

import org.w3c.dom.Element;

import com.wolfram.bsf.util.concurrent.InvokeRunnable;
import com.wolfram.guikit.type.GUIKitTypedObject;

import com.wolfram.guikit.GUIKitEnvironment;
import com.wolfram.guikit.GUIKitException;

/**
 * InvokeProcessScriptBeanRunnable
 */
public class InvokeProcessScriptBeanRunnable extends InvokeRunnable {
   
    private GUIKitEnvironment env;
    private GUIKitTypedObject context;
    private URL contextURL;
    private Element script;
     
    public InvokeProcessScriptBeanRunnable(GUIKitEnvironment env, GUIKitTypedObject context, URL contextURL, Element script) {
            
      this.env = env;
      this.context = context;
      this.contextURL = contextURL;
      this.script = script;
      }
      
    public void run() {
      
      try {
         env.getDriver().processScriptBean( env, script, context, contextURL);
        }
      catch (Exception e) {
        exception = e;
        }
      
      cleanup();
      }
      
	public void cleanup() {
      this.env = null;
      this.context = null;
      this.contextURL = null;
      this.script = null;
      }
      
	public void handleException() throws Exception {
		if (exception != null) {
			if (exception instanceof GUIKitException)
				throw (GUIKitException)exception;
			}
		}
		
	}