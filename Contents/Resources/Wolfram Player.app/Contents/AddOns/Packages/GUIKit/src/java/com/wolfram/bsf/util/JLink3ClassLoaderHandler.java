/*
 * @(#)JLink3ClassLoaderHandler.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

import com.wolfram.jlink.JLinkClassLoader;

/**
 * JLink3ClassLoaderHandler
 */
public class JLink3ClassLoaderHandler implements LinkClassLoaderHandler {
  
  public JLink3ClassLoaderHandler() {
    }
  
  public ClassLoader getClassLoader() {
    return JLinkClassLoader.getInstance();
    }
  
  public Class classFromBytes(String className, byte[] bytes) {
    return JLinkClassLoader.getInstance().classFromBytes(className, bytes);
    }
  
  public String[] getClassPath() {
    return JLinkClassLoader.getInstance().getClassPath();
    }
    
  public Class classLoad(String name) throws ClassNotFoundException {
    return JLinkClassLoader.getInstance().loadClass(name);
    }
  
  }