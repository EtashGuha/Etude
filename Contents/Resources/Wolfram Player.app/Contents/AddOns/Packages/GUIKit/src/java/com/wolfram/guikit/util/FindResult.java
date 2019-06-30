  /*
 * @(#)KeyUtils.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.util;

import javax.swing.Action;
import javax.swing.InputMap;
import javax.swing.text.Keymap;

public class FindResult {
    // Non-null if the keystroke is in an inputmap
    public InputMap inputMap;

    // Non-null if the keystroke is in a keymap or default action
    public Keymap keymap;

    // Non-null if the keystroke is in a default action
    // The keymap field holds the keymap containing the default action
    public Action defaultAction;

    // If true, the keystroke is in the component's inputMap or keymap
    // and not in one of the inputMap's or keymap's parent.
    public boolean isLocal;

    public String toString() {
      StringBuffer b = new StringBuffer();

      b.append("inputmap="+inputMap+",keymap="+keymap
        +",defaultAction="+defaultAction+",isLocal="+isLocal);
      return b.toString();
      }
    }