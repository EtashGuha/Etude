/*
 * @(#)InvokeFieldRunnable.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.concurrent;

import java.lang.reflect.Field;

/**
 * InvokeFieldRunnable
 */
public class InvokeFieldRunnable extends InvokeRunnable {
    private Field field;
    private Object obj;
    private Object val;
    
    public InvokeFieldRunnable(Field f, Object o, Object v) {
      field = f;
      obj = o;
      val = v;
      }
      
    public void run() {
      try {
        field.set(obj, val);
        }
      catch (IllegalAccessException ie) {
        exception = ie;
        }
      cleanup();
      }
      
    public void cleanup() {
      field = null;
      obj = null;
      val = null;
      }
    }