/*
 * @(#)EvalPacketListener.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.engines;

import com.wolfram.jlink.*;

import java.io.PrintStream;

/**
 * EvalPacketListener
 *
 * A default PacketListener to hook into the MathematicaBSFEngine to at
 * least print out any TextPacket[] and MessagePacket[] content to System.err
 *
 * Eventually add a mechanism to allow a user to add/change this listener
 */
public class EvalPacketListener implements PacketListener {

	private boolean lastWasMessage = false;
  private PrintStream stream;

  public EvalPacketListener(PrintStream stream) {
    this.stream = stream;
    }

  public void reset() {
    lastWasMessage = false;
    }

  public boolean packetArrived(PacketArrivedEvent evt) throws MathLinkException {
    KernelLink ml = (KernelLink)evt.getSource();
    switch (evt.getPktType()) {
      case MathLink.MESSAGEPKT:
      case MathLink.TEXTPKT: {
        String s = ml.getString();
        handleString(evt.getPktType(), s, lastWasMessage);
        break;
        }
      case MathLink.EXPRESSIONPKT: {
        String s = ml.getExpr().toString();
        handleString(evt.getPktType(), s, lastWasMessage);
        break;
        }
      default:
        break;
      }
    lastWasMessage = (evt.getPktType() == MathLink.MESSAGEPKT);
    return true;
    }

  private void handleString(final int pkt, final String s, final boolean lastWasMessage) {
    String msg = s;
    switch (pkt) {
      case MathLink.EXPRESSIONPKT:
      case MathLink.TEXTPKT: {
        if (lastWasMessage) {
          if (!msg.endsWith("\n"))
            msg = msg + "\n";
          if (!msg.endsWith("\n\n"))
            msg = msg + "\n";
          }
        else {
          stream.print("\n[MathematicaBSFEngine] Text :\n");
          }
        stream.print(msg);
        break;
        }
      case MathLink.MESSAGEPKT:
        // Don't insert the string--the actual text comes in the subsequent TextPacket.
        stream.print("\n[MathematicaBSFEngine] Message :\n");
        break;
      }
    }

}
