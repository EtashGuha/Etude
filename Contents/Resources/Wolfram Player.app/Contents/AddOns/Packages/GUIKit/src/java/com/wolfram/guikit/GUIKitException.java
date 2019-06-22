/*
 * @(#)GUIKitException.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit;

/**
 * GUIKitException
 */
public class GUIKitException extends Exception {

  private static final long serialVersionUID = -1287987975496789948L;
    
  public static int REASON_INVALID_ARGUMENT;
  public static int REASON_UNKNOWN_CLASS = 10;
  public static int REASON_UNKNOWN_OBJECT = 11;
  public static int REASON_UNKNOWN_TYPECVTOR = 12;
  public static int REASON_UNKNOWN_SIGCVTOR = 13;
  public static int REASON_UNKNOWN_ADDER = 14;
  public static int REASON_UNKNOWN_EVENTADAPTER = 15;
  public static int REASON_GET_METHOD_ERROR = 20;
  public static int REASON_GET_CONSTRUCTOR_ERROR = 21;
  public static int REASON_GET_OBJECTINFO_ERROR = 22;
  public static int REASON_CREATE_OBJECT_ERROR = 25;
  public static int REASON_CALL_METHOD_ERROR = 26;
  public static int REASON_OTHER_ERROR = 50;

  private int reason;
  private Throwable targetThrowable;

  public GUIKitException(int i, String s, Throwable throwable) {
    this(i, s);
    targetThrowable = throwable;
    }

  public GUIKitException(int i, String s) {
    super(s);
    reason = i;
    }

  public int getReason() {
    return reason;
    }

  public Throwable getTargetException() {
    return targetThrowable;
    }

}
