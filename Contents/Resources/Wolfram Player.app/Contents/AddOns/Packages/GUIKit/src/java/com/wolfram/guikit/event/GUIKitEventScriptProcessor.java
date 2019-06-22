/*
 * @(#)GUIKitEventScriptProcessor.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.event;

import com.wolfram.guikit.*;
import com.wolfram.guikit.type.GUIKitTypedObject;
import com.wolfram.guikit.concurrent.*;

import com.wolfram.bsf.util.concurrent.InvokeMode;
import com.wolfram.bsf.util.type.TypedObjectFactory;
import com.wolfram.bsf.util.MathematicaBSFManager;
import com.wolfram.bsf.util.MathematicaEngineUtils;
import com.wolfram.bsf.util.MathematicaObjectRegistry;

import java.io.IOException;
import java.net.URL;

import javax.swing.SwingUtilities;

import org.w3c.dom.Element;

/**
 * GUIKitEventScriptProcessor
 */
public class GUIKitEventScriptProcessor extends GUIKitEventProcessor {

  protected String scriptString;
  protected String scriptSource;
  protected String language;

  public GUIKitEventScriptProcessor() {
    super();
    }

  public Object process(String evalContext) {
    // Currently we assume only using scriptString <script> bindevent
    // and assume Mathematica language.
    // see if it is possible to support the GUI_XMLFORMAT version?? or alternate script
    // Perhaps not especially if using Mathematica results, but maybe not??
    Object result = null;
        
    if (mode.isCurrentThread() || (mode.isDispatchThread() && SwingUtilities.isEventDispatchThread())) {
      try {
        result = env.evaluateScript(evalContext, scriptString);
        }
      catch (GUIKitException be) {
         be.printStackTrace();
         }
      }
    else {
    	try {
      	result = InvokeMode.processResult(mode, new InvokeEngineEvaluateRunnable(env, evalContext, scriptString));
    		}
    	catch (Exception e) {}
      }
    return result;
    }
    
  public void setRootElement(Element element) throws GUIKitException {
    super.setRootElement(element);
    
    language = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_LANGUAGE);

    scriptSource = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_SRC);

    if(language == null) {
      language = MathematicaBSFManager.MATHEMATICA_LANGUAGE_NAME;
      }
    if(!language.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_GUI_XMLFORMAT) && 
       !language.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_XML))
      scriptString = GUIKitUtils.getChildCharacterData(element);
    }


  public void processExceptionableEvent(String eventName, Object aobj[]) throws Exception {
    if(filter != null && !filter.equalsIgnoreCase(eventName)) return;

    // If we have a src attribute need to use this external content for script
    if (scriptSource != null) {
      URL srcURL = null;
      srcURL = MathematicaEngineUtils.getMathematicaURL(GUIKitEnvironment.RESOLVE_FUNCTION,
        contextURL, scriptSource, env.getMathematicaEngine(), env.getDriver().getURLCache());
      if (srcURL == null) srcURL = contextURL;

      if(language.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_GUI_XMLFORMAT) || 
         language.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_XML)) {
        // This may not expose subelement ids when in an external file than when explicitly
        // in the document. Need to see if this is an issue
        if (mode.isCurrentThread() || (mode.isDispatchThread() && SwingUtilities.isEventDispatchThread())) {
          env.getDriver().processScriptBean( env, rootElement, context, srcURL);
          }
        else {   
          InvokeMode.process(mode, new InvokeProcessScriptBeanRunnable(env, context, srcURL, rootElement));
          }
        }
      else {
        // needs to be contents of file
        // check if the protocol is file and evaluate the contents through
        //  a call to Get[] and not pulling in the String contents
        Object scriptContent = null;
        try {
          if (srcURL != null && "file".equals(srcURL.getProtocol())) {
            scriptContent = srcURL;
            //System.out.println("File Get EventScriptProcessor: " + srcURL.getPath());
            }
          else {
            scriptContent = MathematicaEngineUtils.getContentAsString(srcURL);
            }
          }
        catch (IOException ie) {
          ie.printStackTrace();
          }
          
				if (mode.isCurrentThread() || (mode.isDispatchThread() && SwingUtilities.isEventDispatchThread())) {
              
          if(context != null) {
            env.pushObjectRegistry(MathematicaObjectRegistry.SCOPE_ACTION, true);
            }
          try {   
            // This should be an ACTION scope and technically can be cleared out when done
            // and this should happen with the pop below
            GUIKitUtils.registerAsScopeArguments(env, eventName, 
             (GUIKitTypedObject[])TypedObjectFactory.createTypedArray(aobj), MathematicaObjectRegistry.SCOPE_ACTION);
            
            GUIKitUtils.evaluateBSFScript(env, context, srcURL, null, language, scriptContent);
            }
          finally {
            if(context != null)
              env.popObjectRegistry(true);
            }
          }
        else {
          InvokeMode.process(mode, 
						new InvokeEvaluateScriptRunnable(env, context, srcURL, null, null, null, scriptContent, language));
          }
        }

      }
    else {

      try {
        
        if (mode.isCurrentThread() || 
          (mode.isDispatchThread() && SwingUtilities.isEventDispatchThread())) {
              
          env.pushObjectRegistry(MathematicaObjectRegistry.SCOPE_ACTION, true);
          try {
            // This should be an ACTION scope and technically can be cleared out when done
            // and this should happen with the pop below
            GUIKitUtils.registerAsScopeArguments(env, eventName, 
              (GUIKitTypedObject[])TypedObjectFactory.createTypedArray(aobj), MathematicaObjectRegistry.SCOPE_ACTION);
            
            if(language.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_GUI_XMLFORMAT) || 
               language.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_XML)) {
              GUIKitUtils.evaluateGUIKitElement(env, context, contextURL, rootElement, MathematicaObjectRegistry.SCOPE_ACTION);
              }
            else {
              GUIKitUtils.evaluateBSFScript(env, context, contextURL, null, language, scriptString);
              }
            }
          finally {
            env.popObjectRegistry(true);
            }
          
          }
        else {
          InvokeEvaluateScriptRunnable r = new InvokeEvaluateScriptRunnable(
              env, context, contextURL, eventName, (GUIKitTypedObject[])TypedObjectFactory.createTypedArray(aobj), 
              rootElement, scriptString, language);
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

}
