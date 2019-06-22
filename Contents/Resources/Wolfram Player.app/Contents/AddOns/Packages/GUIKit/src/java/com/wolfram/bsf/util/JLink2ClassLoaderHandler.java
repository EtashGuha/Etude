/*
 * @(#)JLink2ClassLoaderHandler.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import com.wolfram.jlink.JLinkClassLoader;

/**
 * JLink2ClassLoaderHandler
 */
public class JLink2ClassLoaderHandler implements LinkClassLoaderHandler {
  
  private static Method getClassLoaderMethod = null;
  private static Method classFromBytesMethod = null;
  private static Method getClassPathMethod = null;
  private static Method classLoadMethod = null;
       
  private static final Object[] NULL_ARRAY = new Object[]{};
  
  public JLink2ClassLoaderHandler() {
    if (getClassLoaderMethod == null) {
      getClassLoaderMethod = MathematicaMethodUtils.getMatchingAccessibleMethod(JLinkClassLoader.class, "getInstance", new Class[]{});
      classFromBytesMethod = MathematicaMethodUtils.getMatchingAccessibleMethod(JLinkClassLoader.class, "classFromBytes", 
          new Class[]{String.class, byte[].class});
      getClassPathMethod = MathematicaMethodUtils.getMatchingAccessibleMethod(JLinkClassLoader.class, "getClassPath", new Class[]{});
      classLoadMethod = MathematicaMethodUtils.getMatchingAccessibleMethod(JLinkClassLoader.class, "load", new Class[]{String.class});
      }
    }
  
  public ClassLoader getClassLoader() {
    try {
      return (ClassLoader)getClassLoaderMethod.invoke(null, NULL_ARRAY);
      }
    catch (IllegalAccessException ex) {ex.printStackTrace();}
    catch (InvocationTargetException ex2) {ex2.printStackTrace();}
    return null;
    }
  
  public Class classFromBytes(String className, byte[] bytes) {
    try {
      return (Class)classFromBytesMethod.invoke(null, new Object[]{className, bytes});
      }
    catch (IllegalAccessException ex) {ex.printStackTrace();}
    catch (InvocationTargetException ex2) {ex2.printStackTrace();}
    return null;
    }
  
  public String[] getClassPath() {
    try {
      return (String[])getClassPathMethod.invoke(null, NULL_ARRAY);
      }
    catch (IllegalAccessException ex) {ex.printStackTrace();}
    catch (InvocationTargetException ex2) {ex2.printStackTrace();}
    return null;
    }
    
  public Class classLoad(String name) throws ClassNotFoundException {
    try {
      return (Class)classLoadMethod.invoke(null, new Object[]{name});
      }
    catch (IllegalAccessException ex) {ex.printStackTrace();}
    catch (InvocationTargetException ex2) {ex2.printStackTrace();}
    return null;
    }
  
  }