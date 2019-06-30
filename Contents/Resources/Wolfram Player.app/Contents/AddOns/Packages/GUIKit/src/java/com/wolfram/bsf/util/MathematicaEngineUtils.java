/*
 * @(#)MathematicaEngineUtils.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

// BSF import switch
import org.apache.bsf.util.StringUtils;
//

import java.net.URL;
import java.net.MalformedURLException;
import java.util.Hashtable;
import java.util.Stack;
import java.io.*;

import com.wolfram.bsf.engines.MathematicaBSFEngine;

/**
 * This class contains utilities that language integrators can use
 * when implementing the BSFEngine interface.
 */
public class MathematicaEngineUtils {

  public static String ATTVAL_PROPERTY_CHANGE = "propertyChange";
  public static String ATTVAL_VETOABLE_CHANGE = "vetoableChange";

  public static String getStringFromReader (Reader reader) throws IOException {
    BufferedReader bufIn = new BufferedReader(reader);
    StringWriter   swOut = new StringWriter();
    PrintWriter    pwOut = new PrintWriter(swOut);
    String         tempLine;
    while ((tempLine = bufIn.readLine()) != null) {
      pwOut.println(tempLine);
      }
    pwOut.flush();
    return swOut.toString();
    }
  
  public static String getContentAsString(URL url) 
      throws SecurityException, IllegalArgumentException, IOException {
    return getStringFromReader( StringUtils.getContentAsReader(url));
    }

  private static boolean canOpenStream(URL url) {
    boolean result = false;
    try {
      url.openStream();
      result = true;
      }
    catch (IOException ioe1) {}
    return result;
    }

  private static URL handleOneURLResolve(URL contextURL, String spec, String convertedSpec) {
    URL url = null;
    try {
    if (contextURL != null && contextURL.getProtocol().equals("file")) {
      url = new URL(contextURL, convertedSpec);
      if (!canOpenStream(url)) {
        url = null;
        }
      }
    else {
      url = new URL(contextURL, spec);
      if (!canOpenStream(url)) {
        url = null;
        URL newurl = new URL(contextURL, spec + ".xml");
        if (!canOpenStream(newurl)) {
          URL newurl2 = new URL(contextURL, spec + ".m");
          if (canOpenStream(newurl2)) {
             url = newurl2;
            }
          }
        else url = newurl;
        }
      }
    }
    catch (MalformedURLException e) {url = null;}
    return url;
    }
  
  public static URL getMathematicaURL(String resolveFunction, URL contextURL, String spec,
      MathematicaBSFEngine mathEngine, Hashtable urlCache) {
    URL result = null;
    result = getMathematicaURL(resolveFunction, contextURL, spec, mathEngine, null, urlCache);
    if (result != null && mathEngine != null && mathEngine.getMathematicaBSFManager() != null && 
        mathEngine.getMathematicaBSFManager().getDebug())
      mathEngine.getMathematicaBSFManager().getDebugStream().println("getMathematicaURL resolved: " + result.toExternalForm());
    return result;
    }
    
  private static URL getMathematicaURL(String resolveFunction, URL contextURL, String spec,
       MathematicaBSFEngine mathEngine, String origSpec, Hashtable urlCache) {
    if (spec == null) return null;
    URL url = null;
    String convertedSpec = spec.trim();
    boolean urlString = false;
    
    if (convertedSpec.startsWith("http:") || convertedSpec.startsWith("https:") ||
        convertedSpec.startsWith("file:"))
      urlString = true;
      
    // Need to make sure context and relative file use same separator
    // So check if Windows and make '/' '\\'
    if (!urlString && com.wolfram.jlink.Utils.isWindows())
      convertedSpec = convertedSpec.replace('/', '\\');

    url = handleOneURLResolve(contextURL, spec, convertedSpec);
    if (url != null) return url;
    
    // Now we try the document URL stack list
    Stack pathStack = null;
    Object scriptPathStack = null;
    if (mathEngine != null && mathEngine.getMathematicaBSFManager() != null)
       scriptPathStack = mathEngine.getMathematicaBSFManager().lookupBean(MathematicaBSFEngine.ID_SCRIPTURLSTACK);
    if (scriptPathStack != null && scriptPathStack instanceof Stack) {
      pathStack = (Stack)scriptPathStack;
      int stackSize = pathStack.size()-1;
      for (int i = stackSize; i >= 0; --i) {
        Object thisURL = pathStack.elementAt(i);
        if (thisURL != null && thisURL instanceof URL)
          url = handleOneURLResolve((URL)thisURL, spec, convertedSpec);
          if (url != null) return url;
          }
      url = null;
      }
    
    // Now we fall back to complete local file paths
    // If a URL string we avoid the local filesystem checks
    if (urlString) return url;
    
    try {
      url = new URL("file", "", convertedSpec);
      url.openStream();
      }
    catch (Exception ioe2) {
      url = null;
      String parentName = null;

      if (contextURL != null && contextURL.getProtocol().equals("file")) {
        String contextFileName = contextURL.getFile();
        parentName = new File(contextFileName).getParent();

        if (parentName != null) {
          url = getMathematicaURL(resolveFunction, null, new File(parentName, convertedSpec).getAbsolutePath(),
              mathEngine, convertedSpec, urlCache );
          if (url != null) return url;
          }
        }


      // Before going to Mathematica, check class loader Resource mechanism
      ClassLoader ldr = (MathematicaEngineUtils.class).getClassLoader();
      url = ldr.getResource(spec);
      if (url != null) return url;
      url = ldr.getResource(spec + ".xml");
      if (url != null) return url;
      url = ldr.getResource(spec + ".m");
      if (url != null) return url;
      
      // Before we throw file not found, last check is to make
      // a call to Mathematica and possibly find the origSpec file on a kernel path
      String resolvedFile = null;
      String useName = (origSpec != null ? origSpec : convertedSpec);

      String keyName = useName;
      if (parentName != null) {
        keyName = "{" + parentName + "," + useName + "}";
        }

      // Before calling Mathematica check urlCache, but also store elements in there
      // when found below.
      // Consider using urlCache everywhere here with keys that wrap contexts too

      if (urlCache != null) {
        resolvedFile = (String)urlCache.get(keyName);
        if (resolvedFile != null && mathEngine != null) {
          mathEngine.log("Using cached resolvedFile for :" + keyName);
          }
        }

      if (resolvedFile == null && mathEngine != null)
        resolvedFile = mathEngine.resolveMathematicaFile(resolveFunction, useName, parentName);

      if (resolvedFile != null) {
        String convertedResolvedFile = resolvedFile;
        if (com.wolfram.jlink.Utils.isWindows())
          convertedResolvedFile = convertedResolvedFile.replace('/', '\\');

        try {
          url = new URL("file", "", convertedResolvedFile);
          url.openStream();
          // If successful add this to the cache of resolved files
          // TODO come up with a mechanism when cache is cleared
          // or not used if needed
          if (urlCache != null)
            urlCache.put(keyName, resolvedFile);
          return url;
          }
        catch (Exception ioe1) {
          url = null;
          }
        }

      }

    return url;
    }

}
