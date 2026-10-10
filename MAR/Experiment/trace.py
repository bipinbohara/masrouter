"""Opt-in observation of graph execution; no routing or generation decisions."""
from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime, timezone
import time

_collector = ContextVar('experiment_collector', default=None)
_call = ContextVar('experiment_call', default=None)
_execution = ContextVar('experiment_execution', default=None)
_phase = ContextVar('experiment_phase', default=None)

def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def collect_traces():
    traces = []
    token = _collector.set(traces)
    try:
        yield traces
    finally:
        _collector.reset(token)


def record_usage(usage):
    call = _call.get()
    if call is not None and usage is not None:
        call['usage'] = {'prompt_tokens': usage.prompt_tokens,
                         'completion_tokens': usage.completion_tokens,
                         'total_tokens': usage.total_tokens}


def record_parameters(request):
    call = _call.get()
    if call is not None:
        call['parameters'] = {key: value for key, value in request.items()
                              if key not in ('messages', 'model')}


def instrument_client_kwargs(kwargs, asynchronous=False):
    call = _call.get()
    if call is None:
        return kwargs
    from openai import DefaultHttpxClient, DefaultAsyncHttpxClient
    call['http_requests'] = 0
    def requested(request):
        call['http_requests'] += 1
    async def async_requested(request):
        requested(request)
    client = (DefaultAsyncHttpxClient(event_hooks={'request': [async_requested]}) if asynchronous
              else DefaultHttpxClient(event_hooks={'request': [requested]}))
    return dict(kwargs, http_client=client)


def run_graph(graph, inputs, num_rounds, routing=None):
    traces = _collector.get()
    if traces is None:
        return graph.run(inputs=inputs, num_rounds=num_rounds)
    trace = {'started_at': utc_now(), 'collaboration': graph.reasoning_name,
             'num_rounds': num_rounds, 'agents': [], 'calls': [], 'events': [], 'rounds': [], 'status': 'running'}
    trace['routing'] = routing() if callable(routing) else routing
    traces.append(trace)
    restorations = []
    for index, node in enumerate([*graph.nodes.values(), graph.decision_node]):
        final = node is graph.decision_node
        agent = {'agent_id': node.id, 'agent_index': index,
                 'role': 'FinalRefer' if final else node.role.role,
                 'model': node.llm.model_name, 'is_final_node': final}
        agent.update(post_process=getattr(node, 'post_process', None),
                     post_output_format=getattr(node, 'post_output_format', None),
                     reasoning=getattr(node, 'reason', None))
        trace['agents'].append(agent)
        for method_name in ('gen', 'agen'):
            llm = node.llm
            original = getattr(llm, method_name)
            existed = method_name in vars(llm)
            previous = vars(llm).get(method_name)
            restorations.append((llm, method_name, existed, previous))
            def begin(messages, kwargs, agent=agent):
                call = dict(agent, call_index=len(trace['calls']), started_at=utc_now(),
                            messages=messages, parameters=kwargs, status='running',
                            usage=None, http_requests=None)
                execution = _execution.get() or {}
                call.update(round_index=execution.get('round_index'),
                            node_attempt=execution.get('node_attempt'),
                            execution_id=execution.get('execution_id'),
                            phase=_phase.get() or ('final_aggregation' if agent['is_final_node'] else 'unclassified'))
                trace['calls'].append(call)
                return call
            def synchronous(messages, *args, original=original, begin=begin, **kwargs):
                call = begin(messages, kwargs)
                token = _call.set(call)
                start = time.perf_counter()
                try:
                    response = original(messages, *args, **kwargs)
                    call.update(status='success', response=response)
                    return response
                except BaseException as error:
                    call.update(status='error', error=f'{type(error).__name__}: {error}')
                    raise
                finally:
                    call['seconds'] = time.perf_counter() - start
                    _call.reset(token)
            async def asynchronous(messages, *args, original=original, begin=begin, **kwargs):
                call = begin(messages, kwargs)
                token = _call.set(call)
                start = time.perf_counter()
                try:
                    response = await original(messages, *args, **kwargs)
                    call.update(status='success', response=response)
                    return response
                except BaseException as error:
                    call.update(status='error', error=f'{type(error).__name__}: {error}')
                    raise
                finally:
                    call['seconds'] = time.perf_counter() - start
                    _call.reset(token)
            setattr(llm, method_name, asynchronous if method_name == 'agen' else synchronous)
    start = time.perf_counter()
    try:
        result = graph.run(inputs=inputs, num_rounds=num_rounds)
        trace.update(status='success', raw_answer=result[0][0])
        return result
    except BaseException as error:
        trace.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        trace['graph_seconds'] = time.perf_counter() - start
        trace['finished_at'] = utc_now()
        for llm, name, existed, previous in reversed(restorations):
            if existed:
                setattr(llm, name, previous)
            else:
                delattr(llm, name)
        for node, agent in zip([*graph.nodes.values(), graph.decision_node], trace['agents']):
            agent['outputs'] = list(node.outputs)


@contextmanager
def generation_phase(phase):
    token = _phase.set(phase)
    try:
        yield
    finally:
        _phase.reset(token)


def current_trace():
    traces = _collector.get()
    return traces[-1] if traces else None


def record_reuse(source_id, source_kind, tests):
    trace = current_trace()
    if trace is not None:
        trace['events'].append(dict(_execution.get() or {}, event='reuse_validated_answer',
                                    timestamp=utc_now(), source_agent_id=source_id,
                                    source_kind=source_kind, internal_tests=tests))


def record_round(graph, round_index):
    trace = current_trace()
    if trace is None:
        return
    trace['rounds'].append({'round_index': round_index, 'timestamp': utc_now(),
                           'spatial_edges': [[node.id, successor.id] for node in graph.nodes.values()
                                             for successor in node.spatial_successors],
                           'temporal_edges': [[node.id, successor.id] for node in graph.nodes.values()
                                              for successor in node.temporal_successors]})


def execute_node(node, inputs, round_index, attempt=1, final=False):
    trace = current_trace()
    if trace is None:
        return node.execute(inputs)
    event = {'event': 'node_execution', 'execution_id': len(trace['events']),
             'agent_id': node.id, 'round_index': round_index, 'node_attempt': attempt,
             'is_final_node': final, 'started_at': utc_now(), 'status': 'running',
             'spatial_sources': [n.id for n in node.spatial_predecessors],
             'temporal_sources': [n.id for n in node.temporal_predecessors]}
    trace['events'].append(event)
    token = _execution.set(event.copy())
    before = len(trace['calls'])
    start = time.perf_counter()
    try:
        result = node.execute(inputs)
        event.update(status='success', outputs=list(node.outputs))
        return result
    except BaseException as error:
        event.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        event.update(seconds=time.perf_counter()-start, finished_at=utc_now(),
                     llm_calls=len(trace['calls'])-before)
        _execution.reset(token)


@contextmanager
def processing_stage(stage):
    trace = current_trace()
    if trace is None:
        yield
        return
    event = dict(_execution.get() or {}, event='processing_stage', stage=stage,
                 started_at=utc_now(), status='running')
    trace['events'].append(event)
    start = time.perf_counter()
    try:
        yield
        event['status'] = 'success'
    except BaseException as error:
        event.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        event.update(seconds=time.perf_counter()-start, finished_at=utc_now())

