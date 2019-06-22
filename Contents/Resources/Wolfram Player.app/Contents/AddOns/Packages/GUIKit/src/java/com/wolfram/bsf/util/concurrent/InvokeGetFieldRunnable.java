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
public class InvokeGetFieldRunnable extends InvokeResultRunnable {
    private Field field;
    private Object obj;
    
    public InvokeGetFieldRunnable(Field f, Object o) {
      field = f;
      obj = o;
      }
      
    public void run() {
      Object result = null;
      try {
        result = field.get(obj);
        }
      catch (IllegalAccessException ie) {
        exception = ie;
        }
      finally {
        setResult(result);
        }
      cleanup();
      }
      
    public void cleanup() {
      field = null;
      obj = null;
      }
    }