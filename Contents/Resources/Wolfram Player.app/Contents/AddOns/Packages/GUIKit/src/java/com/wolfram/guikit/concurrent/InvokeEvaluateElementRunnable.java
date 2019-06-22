/*
 * @(#)InvokeEvaluateElementRunnable.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.concurrent;

import java.net.URL;

import org.w3c.dom.Element;

import com.wolfram.guikit.type.GUIKitTypedObject;
import com.wolfram.bsf.util.concurrent.InvokeRunnable;
import com.wolfram.bsf.util.MathematicaObjectRegistry;

import com.wolfram.guikit.GUIKitEnvironment;
import com.wolfram.guikit.GUIKitException;
import com.wolfram.guikit.GUIKitUtils;

/**
 * InvokeEvaluateElementRunnable
 */
public class InvokeEvaluateElementRunnable extends InvokeRunnable {
   
    protected GUIKitEnvironment env;
    protected GUIKitTypedObject context;
    protected URL contextURL;
    protected String eventName;
    protected GUIKitTypedObject args[];
    protected Element rootElement;
     
    public InvokeEvaluateElementRunnable(GUIKitEnvironment env, GUIKitTypedObject context, URL contextURL, 
      String eventName, GUIKitTypedObject args[], Element rootElement) {
            
      this.env = env;
      this.context = context;
      this.contextURL = contextURL;
      this.eventName = eventName;
      this.args = args;
      this.rootElement = rootElement;
      }
      
    public void run() {
      
      env.pushObjectRegistry(MathematicaObjectRegistry.SCOPE_ACTION, true);
        
      try {
        // This should be an ACTION scope and technically can be cleared out when done
        // and this should happen with the pop below
        GUIKitUtils.registerAsScopeArguments(env, eventName, args, MathematicaObjectRegistry.SCOPE_ACTION);
        GUIKitUtils.evaluateGUIKitElement(env, context, contextURL, rootElement, MathematicaObjectRegistry.SCOPE_ACTION);
        }
      catch (Exception e) {
        exception = e;
        }
      finally {
        env.popObjectRegistry(true);
        }
      
      cleanup();
      }
      
    public void cleanup() {
      this.env = null;
      this.context = null;
      this.contextURL = null;
      this.eventName = null;
      this.args = null;
      this.rootElement = null;
      }
    
	public void handleException() throws Exception {
		if (exception != null) {
			if (exception instanceof GUIKitException)
				throw (GUIKitException)exception;
			}
		}
		
    }