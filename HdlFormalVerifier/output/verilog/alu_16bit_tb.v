//==============================================================================
// Testbench: alu_16bit
// Generated: 2025-12-05 19:51:16
// Generator: testbench_generator.py (deterministic, no LLM)
//
// DUT Module: alu_16bit
// Bitwidth: 16-bit
// Test cases: 20
// Number format: decimal
//==============================================================================

`timescale 1ns / 1ps

module alu_16bit_tb;

    //--------------------------------------------------------------------------
    // Test Signals
    //--------------------------------------------------------------------------
    reg  [15:0] a;
    reg  [15:0] b;
    reg  [3:0] opcode;
    wire [15:0] result;
    wire zero, carry, overflow, negative;

    // Test counters
    integer passed = 0;
    integer failed = 0;
    integer total = 0;

    //--------------------------------------------------------------------------
    // Device Under Test (DUT)
    //--------------------------------------------------------------------------
    alu_16bit uut (
        .a(a),
        .b(b),
        .opcode(opcode),
        .result(result),
        .zero(zero),
        .carry(carry),
        .overflow(overflow),
        .negative(negative)
    );

    //--------------------------------------------------------------------------
    // Test Sequence
    //--------------------------------------------------------------------------
    initial begin
        $display("========================================");
        $display("Testbench: alu_16bit");
        $display("DUT: alu_16bit");
        $display("========================================");
        $display("");

        // Test 1: ADD
        a = 16'd10;
        b = 16'd5;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd15) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd15);
            failed = failed + 1;
        end

        // Test 2: ADD
        a = 16'd118;
        b = 16'd118;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd236) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd236);
            failed = failed + 1;
        end

        // Test 3: ADD
        a = 16'd92;
        b = 16'd0;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd92) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd92);
            failed = failed + 1;
        end

        // Test 4: ADD
        a = 16'd21338;
        b = 16'd27547;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd48885) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd48885);
            failed = failed + 1;
        end

        // Test 5: ADD
        a = 16'd240;
        b = 16'd18533;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd18773) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd18773);
            failed = failed + 1;
        end

        // Test 6: SUB
        a = 16'd10;
        b = 16'd5;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd5) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd5);
            failed = failed + 1;
        end

        // Test 7: SUB
        a = 16'd400;
        b = 16'd400;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd0);
            failed = failed + 1;
        end

        // Test 8: SUB
        a = 16'd34;
        b = 16'd0;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd34) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd34);
            failed = failed + 1;
        end

        // Test 9: SUB
        a = 16'd3773;
        b = 16'd26952;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd42357) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd42357);
            failed = failed + 1;
        end

        // Test 10: SUB
        a = 16'd31490;
        b = 16'd15021;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd16469) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd16469);
            failed = failed + 1;
        end

        // Test 11: AND
        a = 16'd10;
        b = 16'd5;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd0);
            failed = failed + 1;
        end

        // Test 12: AND
        a = 16'd628;
        b = 16'd628;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd628) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd628);
            failed = failed + 1;
        end

        // Test 13: AND
        a = 16'd46;
        b = 16'd0;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd0);
            failed = failed + 1;
        end

        // Test 14: AND
        a = 16'd15865;
        b = 16'd1376;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd1376) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd1376);
            failed = failed + 1;
        end

        // Test 15: AND
        a = 16'd20554;
        b = 16'd22473;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd20552) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd20552);
            failed = failed + 1;
        end

        // Test 16: OR
        a = 16'd10;
        b = 16'd5;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd15) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd15);
            failed = failed + 1;
        end

        // Test 17: OR
        a = 16'd580;
        b = 16'd580;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd580) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd580);
            failed = failed + 1;
        end

        // Test 18: OR
        a = 16'd27;
        b = 16'd0;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd27) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd27);
            failed = failed + 1;
        end

        // Test 19: OR
        a = 16'd11011;
        b = 16'd15044;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd15303) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd15303);
            failed = failed + 1;
        end

        // Test 20: OR
        a = 16'd31452;
        b = 16'd2244;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 16'd31452) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 16'd31452);
            failed = failed + 1;
        end

        $display("");
        $display("========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Total:  %0d", total);
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);
        if (failed == 0)
            $display("[SUCCESS] All tests passed!");
        else
            $display("[FAILURE] Some tests failed!");
        $display("========================================");

        #10;
        $finish;
    end

    //--------------------------------------------------------------------------
    // VCD Dump for Waveform Viewing
    //--------------------------------------------------------------------------
    initial begin
        $dumpfile("alu_16bit_tb.vcd");
        $dumpvars(0, alu_16bit_tb);
    end

endmodule

//==============================================================================
// End of Testbench
//==============================================================================