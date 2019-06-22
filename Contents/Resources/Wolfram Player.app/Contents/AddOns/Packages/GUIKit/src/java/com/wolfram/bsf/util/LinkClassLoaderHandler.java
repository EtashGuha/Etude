/*
 * @(#)LinkClassLoaderHandler.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

/**
 * LinkClassLoaderHandler
 */
public interface LinkClassLoaderHandler {
  
  public ClassLoader getClassLoader();
  
  public Class classFromBytes(String className, byte[] bytes);
  
  public String[] getClassPath();
    
  public Class classLoad(String name) throws ClassNotFoundException;
  
  }