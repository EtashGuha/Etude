/*
 * @(#)GUIKitEventProcessor.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.event;

import com.wolfram.guikit.*;
import com.wolfram.guikit.concurrent.InvokeEvaluateElementRunnable;

import com.wolfram.bsf.util.concurrent.InvokeMode;

import com.wolfram.guikit.type.GUIKitTypedObject;

import com.wolfram.bsf.util.type.TypedObjectFactory;
import com.wolfram.bsf.util.event.MathematicaEventProcessor;
import com.wolfram.bsf.util.MathematicaObjectRegistry;

import java.net.URL;

import javax.swing.SwingUtilities;

import org.w3c.dom.Element;

/**
 * GUIKitEventProcessor
 */
public class GUIKitEventProcessor implements MathematicaEventProcessor, GUIKitEnvironmentObject {

  protected GUIKitEnvironment env;
  protected String filter;
  protected GUIKitTypedObject context;
  protected URL contextURL;
  protected Element rootElement;
  protected InvokeMode mode = InvokeMode.INVOKE_CURRENT;
  
  public GUIKitEventProcessor() {
    }

  public Object process(String evalContext) {
    // This shouldn't be called
    return null;
    }
    
  public void setFilter(String s) {
    filter = s;
    }

  public void setContext(GUIKitTypedObject bean) {
    context = bean;
    }

  public void setContextURL(URL url) {
    contextURL = url;
    }
    
  public void setInvokeMode(InvokeMode m) {
    this.mode = m;
    }
    
  public void setGUIKitEnvironment(GUIKitEnvironment env) {
    this.env = env;
    if (env != null) {
      env.addGUIKitEnvironmentObject(this);
      }
    // we are actually shutting down so cleanup references
    else {
      if (mode != null) mode.setManager(null);
      mode = null;
      context = null;
      contextURL = null;
      rootElement = null;
      }
    }

  public void setRootElement(Element element) throws GUIKitException {
    // TODO think about lifetime issues of holding on to element
    // until event is processed. Think about ways to create any needed
    // content to not have to hold on to element
    rootElement = element;
    }

  public void processEvent(String eventName, Object aobj[]) {
    try {
      processExceptionableEvent(eventName, aobj);
      }
    catch(RuntimeException runtimeexception) {
      throw runtimeexception;
      }
    catch(Exception exception) {
      // Since technically an GUIKitException could be thrown on 'no such method' etc
      // we should try and report this to the user
      // Perhaps these are developer time issues and we should send them a different route
      // or a GUIMessage window
      exception.printStackTrace();
      // Need to see if this can be called when initiated from a Java call not from a Mathematica request
      //  as this case seems to lockup
      //try {
      //  env.getMathematicaEngine().requestHandleException(exception);
      //  }
      //catch (MathLinkException me) {}
      }
    }

  public void processExceptionableEvent(String eventName, Object aobj[]) throws Exception {
    if(filter != null && !filter.equalsIgnoreCase(eventName)) return;
    try {
      
      if (mode.isCurrentThread() || (mode.isDispatchThread() && SwingUtilities.isEventDispatchThread())) {
            
        env.pushObjectRegistry(MathematicaObjectRegistry.SCOPE_ACTION, true);
        try {
          // This should be an ACTION scope and technically can be cleared out when done
          // and this should happen with the pop below
          GUIKitUtils.registerAsScopeArguments(env, eventName, 
            (GUIKitTypedObject[])TypedObjectFactory.createTypedArray(aobj), MathematicaObjectRegistry.SCOPE_ACTION);
          
          GUIKitUtils.evaluateGUIKitElement(env, context, contextURL, rootElement, MathematicaObjectRegistry.SCOPE_ACTION);
          }
        finally {
          env.popObjectRegistry(true);
          }
        }
      else {
        InvokeEvaluateElementRunnable r = new InvokeEvaluateElementRunnable(
            env, context, contextURL, eventName, (GUIKitTypedObject[])TypedObjectFactory.createTypedArray(aobj), 
            rootElement);
        InvokeMode.process(mode, r);
        }
      }
    catch(GUIKitException poexception) {
        Throwable throwable = poexception.getTargetException();
        if(throwable instanceof Exception)
            throw (Exception)throwable;
        else
            throw poexception;
      }
    }

}
