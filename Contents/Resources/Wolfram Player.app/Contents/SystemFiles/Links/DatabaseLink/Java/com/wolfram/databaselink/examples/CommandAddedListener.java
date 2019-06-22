package com.wolfram.databaselink.examples;

import java.util.EventListener;

public interface CommandAddedListener extends EventListener
{
  public void commandAdded(CommandAddedEvent evt);
}

