/*
 * @(#)InvokeEvaluateScriptRunnable.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.concurrent;

import java.net.URL;

import org.w3c.dom.Element;

import com.wolfram.guikit.type.GUIKitTypedObject;
import com.wolfram.bsf.util.MathematicaObjectRegistry;

import com.wolfram.guikit.GUIKitEnvironment;
import com.wolfram.guikit.GUIKitUtils;

/**
 * InvokeEvaluateScriptRunnable
 */
public class InvokeEvaluateScriptRunnable extends InvokeEvaluateElementRunnable {
   
    protected Object scriptContent;
    protected String language;
     
    public InvokeEvaluateScriptRunnable(GUIKitEnvironment env, GUIKitTypedObject context, URL contextURL, 
      String eventName, GUIKitTypedObject args[], Element rootElement, Object scriptContent, String language) {
      super(env, context, contextURL, eventName, args, rootElement);
      this.scriptContent = scriptContent;
      this.language = language;
      }
      
    public void run() {
      
      env.pushObjectRegistry(MathematicaObjectRegistry.SCOPE_ACTION, true);
        
      try {
        // This should be an ACTION scope and technically can be cleared out when done
        // and this should happen with the pop below
        GUIKitUtils.registerAsScopeArguments(env, eventName, args, MathematicaObjectRegistry.SCOPE_ACTION);
        
        if(language.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_GUI_XMLFORMAT) || 
           language.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_XML)) {
          GUIKitUtils.evaluateGUIKitElement(env, context, contextURL, rootElement, MathematicaObjectRegistry.SCOPE_ACTION);
          }
        else {
          GUIKitUtils.evaluateBSFScript(env, context, contextURL, null, language, scriptContent);
          }
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
      super.cleanup();
      this.scriptContent = null;
      this.language = null;
     }
     
    }